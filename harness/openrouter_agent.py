"""The harness's own agent loop, for any tool-calling model on OpenRouter.

    uv run python -m harness.run_agent --model google/gemini-3-pro [--max-cost 20]

Claude runs use Claude Code (harness/run_agent.py). This backend gives other
models the same brief, workspace and harness tools (`submit_policy`, `status`,
`finish`) through OpenRouter's OpenAI-style chat completions API, plus its own
small set of workspace tools:

- `bash`: one command per call, in a bwrap sandbox with no network where the
  workspace is the only writable path (harness/sandbox.py).
- `read_file`, `write_file`, `edit_file`: paths resolve (following symlinks)
  and must stay inside the workspace for writing, or the workspace, `bess/` and
  `harness/` for reading, so the holdout data stays out of reach.

The scaffold differs from Claude Code's, which matters when comparing scores.

Costs are what OpenRouter bills (`usage.cost` on each response), so --max-cost
is a real spending cap. It's checked before every API call as well as by the
watchdog. The conversation is appended to logs/messages.jsonl, which is what a
`--resume` reloads. When the context nears the model's limit, the oldest tool
outputs are elided.
"""

import asyncio
import json
import os
import signal
import time
from pathlib import Path

import httpx

from harness.run_agent import (
    FINISH_DESCRIPTION,
    STATUS_DESCRIPTION,
    SUBMIT_POLICY_DESCRIPTION,
    Run,
    _truncate,
)
from harness.sandbox import PROJECT_ROOT, PersistentSandbox

API_URL = "https://openrouter.ai/api/v1"
KEY_FILE = Path.home() / ".config" / "storage-benchmark" / "openrouter_api_key"
SYSTEM_PROMPT = Path(__file__).resolve().parent / "openrouter_system.md"

OUTPUT_CHARS = 30_000  # tool output kept per call (head and tail)
READ_LIMIT_LINES = 2000
BASH_DEFAULT_S, BASH_MAX_S = 30 * 60, 90 * 60  # as Claude Code's Bash limits in the Claude runs
MAX_OUTPUT_TOKENS = 64_000
TRIM_AT, TRIM_TO, KEEP_RECENT = 0.75, 0.5, 12  # context trimming: fractions of the context window
ELIDED = "[output elided to save context]"
RETRY_DELAYS_S = (5, 30, 120)  # in-call retries for rate limits and provider errors
RESTARTED = "The harness restarted before this tool call finished, so it has no result. Re-run it if you need it."


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        raise SystemExit(f"No OpenRouter key. Export OPENROUTER_API_KEY or save it to {KEY_FILE}.")
    os.environ["OPENROUTER_API_KEY"] = key
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "storage-optimization-benchmark"}


async def model_info(http: httpx.AsyncClient, model: str) -> dict:
    r = await http.get("/models")
    r.raise_for_status()
    info = next((m for m in r.json()["data"] if m["id"] == model), None)
    if info is None:
        raise SystemExit(f"OpenRouter has no model {model!r}. See https://openrouter.ai/models for slugs.")
    if "tools" not in (info.get("supported_parameters") or []):
        raise SystemExit(f"{model} doesn't support tool calling on OpenRouter, which the agent loop needs.")
    return info


async def check_model(model: str, max_cost: float) -> None:
    """Before a run starts: the model exists and supports tools, and the key works."""
    async with httpx.AsyncClient(base_url=API_URL, headers=_headers(), timeout=30) as http:
        info = await model_info(http, model)
        r = await http.get("/key")
        if r.status_code == 401:
            raise SystemExit("OpenRouter rejected the API key (401).")
        r.raise_for_status()
        key = r.json().get("data", {})
    remaining = key.get("limit_remaining")
    print(f"OpenRouter: {model}, context {info.get('context_length'):,} tokens, "
          f"key usage ${key.get('usage', 0):.2f}" + (f", ${remaining:.2f} left on the key" if remaining is not None else ""))
    if remaining is not None and remaining < max_cost:
        print(f"Warning: the key has ${remaining:.2f} left, below --max-cost ${max_cost:.2f}.")


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


TOOLS = [
    _function("bash", (
        "Run a bash command in the workspace (your working directory) and return its output and exit code. Each call "
        "is a fresh shell: `cd` and environment variables don't carry over, and background processes are killed "
        "when the command returns, so run parallel work with `&` and `wait` inside one command. No network. "
        f"Output beyond {OUTPUT_CHARS:,} characters is cut from the middle."
    ), {
        "command": {"type": "string"},
        "timeout_s": {"type": "number", "description": f"default {BASH_DEFAULT_S}, max {BASH_MAX_S}"},
    }, ["command"]),
    _function("read_file", (
        "Read a text file, with line numbers. Relative paths are from the workspace. You can read the workspace and "
        "the `bess` and `harness` package sources."
    ), {
        "path": {"type": "string"},
        "offset": {"type": "integer", "description": "first line to read, from 1"},
        "limit": {"type": "integer", "description": f"number of lines (default {READ_LIMIT_LINES})"},
    }, ["path"]),
    _function("write_file", "Create or overwrite a file in the workspace.", {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, ["path", "content"]),
    _function("edit_file", (
        "Replace an exact string in a workspace file. `old_string` must occur exactly once unless `replace_all` is true."
    ), {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"},
    }, ["path", "old_string", "new_string"]),
    _function("submit_policy", SUBMIT_POLICY_DESCRIPTION, {
        "summary": {"type": "string"}, "rationale": {"type": "string"},
    }, ["summary", "rationale"]),
    _function("status", STATUS_DESCRIPTION, {}, []),
    _function("finish", FINISH_DESCRIPTION, {"summary": {"type": "string"}}, ["summary"]),
]


class ToolError(Exception):
    """A tool call the agent got wrong; the message goes back to it."""


def _clip(text: str, limit: int = OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit:,} characters cut] ...\n{text[-half:]}"


class OpenRouterRun(Run):
    billing = "openrouter"

    def __init__(self, args):
        self.sandbox = None  # the agent's persistent bash sandbox, started on its first command
        super().__init__(args)
        self.messages: list[dict] = []
        self.messages_path = self.dir / "logs" / "messages.jsonl"
        self.http: httpx.AsyncClient | None = None
        self.info: dict | None = None
        self.current: asyncio.Future | None = None  # the API call or tool call in flight
        self.interrupted = False
        self.read_roots = [self.workspace.resolve(), (PROJECT_ROOT / "bess").resolve(), (PROJECT_ROOT / "harness").resolve()]

    # ---- accounting and control ----

    def call_cost(self, usage: dict) -> float:
        return usage.get("cost") or 0.0

    async def interrupt(self) -> None:
        self.interrupted = True
        if self.current is not None:
            self.current.cancel()

    async def cancellable(self, coro):
        """Await `coro`; returns None if `interrupt` cancelled it."""
        self.current = asyncio.ensure_future(coro)
        try:
            return await self.current
        except asyncio.CancelledError:
            if self.interrupted:
                return None
            raise
        finally:
            self.current = None

    # ---- conversation ----

    def append(self, message: dict) -> None:
        self.messages.append(message)
        with self.messages_path.open("a") as f:
            f.write(json.dumps(message) + "\n")

    def restore_conversation(self, state: dict) -> None:
        if self.messages_path.exists():
            self.messages = [json.loads(line) for line in self.messages_path.open() if line.strip()]
        # Tool calls cut off by the restart still need a result each.
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        last = next((m for m in reversed(self.messages) if m.get("role") == "assistant"), None)
        for call in (last or {}).get("tool_calls") or []:
            if call["id"] not in answered:
                self.append({"role": "tool", "tool_call_id": call["id"], "content": RESTARTED})

    def trim_context(self, prompt_tokens: int) -> None:
        """Elide the oldest tool outputs once the prompt nears the context window."""
        window = self.info.get("context_length") or 128_000
        if prompt_tokens < TRIM_AT * window:
            return
        chars = sum(len(json.dumps(m)) for m in self.messages)
        tokens_per_char = prompt_tokens / max(chars, 1)
        estimate, elided = prompt_tokens, 0
        for m in self.messages[:-KEEP_RECENT]:
            if estimate <= TRIM_TO * window:
                break
            if m.get("role") == "tool" and m.get("content") != ELIDED and len(m.get("content", "")) > 200:
                estimate -= (len(m["content"]) - len(ELIDED)) * tokens_per_char
                m["content"] = ELIDED
                elided += 1
        self.log.event("context_trimmed", prompt_tokens=prompt_tokens, context_length=window,
                       elided_tool_outputs=elided, estimate_after=round(estimate))

    # ---- the loop ----

    async def session(self, prompt: str) -> str:
        """Talk to the model until it finishes, a limit is hit, or it stops
        without calling finish too many times. Returns "finished" or "limit"."""
        if self.http is None:
            self.http = httpx.AsyncClient(base_url=API_URL, headers=_headers(),
                                          timeout=httpx.Timeout(900, connect=30))
        if self.info is None:
            self.info = await model_info(self.http, self.model)
            self.log.event("openrouter_model", model=self.model, context_length=self.info.get("context_length"),
                           pricing=self.info.get("pricing"), top_provider=self.info.get("top_provider"))
        if not self.messages:
            self.append({"role": "system", "content": SYSTEM_PROMPT.read_text()})
        self.log.event("prompt", text=prompt)
        self.append({"role": "user", "content": prompt})
        while True:
            reason = self.check_limits()
            if reason and not self.limit_reason:
                self.reach_limit(reason)
            if self.limit_reason:
                return "limit"
            if self.finished_reason:
                return "finished"
            message = await self.cancellable(self.call_model())
            if message is None:  # interrupted, or out of credits
                return "finished" if self.finished_reason else "limit"
            calls = message.get("tool_calls") or []
            for call in calls:
                result = await self.run_tool(call)
                self.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if calls:
                continue
            if self.nudges >= self.args.max_nudges:
                self.finished_reason = "max_nudges"
                return "finished"
            self.nudges += 1
            nudge = ("You ended your turn without calling the finish tool. "
                     f"{self.status_text()} Keep working on the task, or call finish if you are done.")
            self.log.event("prompt", text=nudge)
            self.append({"role": "user", "content": nudge})

    async def call_model(self) -> dict | None:
        """One chat completion. Retries rate limits and provider errors a few
        times with the active clock stopped; raises if they persist (the run
        loop then pauses longer and reconnects)."""
        top = self.info.get("top_provider") or {}
        body = {
            "model": self.model,
            "messages": self.messages,
            "tools": TOOLS,
            "reasoning": {"effort": self.args.effort},
            "max_tokens": min(top.get("max_completion_tokens") or MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS),
        }
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            started = time.time()
            wait_s = started - self.last_message_at
            try:
                r = await self.http.post("/chat/completions", json=body)
                data = r.json() if r.content else {}
            except (httpx.TransportError, json.JSONDecodeError) as e:
                status, data, error = None, {}, f"{type(e).__name__}: {e}"
            else:
                status, error = r.status_code, data.get("error")
                if isinstance(error, dict):
                    error = error.get("message") or json.dumps(error)
            if status == 402:
                self.finished_reason = "no_credits"
                self.log.event("no_credits", detail=error)
                print(f"Stopping: OpenRouter credits exhausted ({error}).", flush=True)
                return None
            if status == 200 and not error and data.get("choices"):
                return self.record_response(data, started, wait_s)
            retryable = status is None or status == 429 or status >= 500 or (status == 200 and not data.get("choices"))
            self.log.event("api_error", status=status, error=_truncate(error or data, 2000), attempt=attempt + 1,
                           retryable=retryable)
            if status == 400 and "context" in str(error).lower():
                self.trim_context(int((self.info.get("context_length") or 128_000) * 0.99))
                continue
            if not retryable or attempt == len(RETRY_DELAYS_S):
                raise RuntimeError(f"OpenRouter error {status}: {error}")
            self.stop_clock()  # waiting out the provider isn't the agent's time
            await asyncio.sleep(RETRY_DELAYS_S[attempt])
            self.start_clock()
        raise RuntimeError("OpenRouter: the context is still too long after trimming")

    def record_response(self, data: dict, started: float, wait_s: float) -> dict:
        now = time.time()
        self.log.transcript(data)
        choice = data["choices"][0]
        message = {k: v for k, v in choice["message"].items() if v is not None}
        message.setdefault("content", "")
        self.append(message)  # passed back as returned, reasoning_details included

        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        written = (u.get("prompt_tokens_details") or {}).get("cache_write_tokens") or 0
        usage = {
            "input_tokens": (u.get("prompt_tokens") or 0) - cached - written,
            "output_tokens": u.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": written,
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "cost": u.get("cost") or 0.0,
        }
        call_id = data.get("id") or f"call-{len(self.api_calls) + 1}"
        self.api_calls[call_id] = usage
        self.log.event(
            "api_call", message_id=call_id, model=data.get("model"), provider=data.get("provider"), usage=usage,
            stop_reason=choice.get("finish_reason"), wait_s=round(wait_s, 3), duration_s=round(now - started, 3),
            call_cost_usd=usage["cost"], cumulative_cost_usd=self.cost_usd(),
        )
        blocks = []
        reasoning = message.get("reasoning") or " ".join(
            d.get("summary") or d.get("text") or "" for d in message.get("reasoning_details") or []
            if d.get("type") in ("reasoning.summary", "reasoning.text"))
        if reasoning.strip():
            blocks.append({"type": "thinking", "text": reasoning})
        if isinstance(message.get("content"), str) and message["content"].strip():
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            blocks.append({"type": "tool_use", "name": fn.get("name"), "input": _truncate(fn.get("arguments", ""), 500)})
        self.log.event("assistant_message", message_id=call_id, model=data.get("model"), new_api_call=True,
                       latency_s=round(now - self.last_message_at, 3), stop_reason=choice.get("finish_reason"),
                       usage=None, blocks=blocks)
        self.last_message_at = now
        self.trim_context(u.get("prompt_tokens") or 0)
        return message

    # ---- tools ----

    async def run_tool(self, call: dict) -> str:
        """Run one tool call and return its result text. Never raises, so every
        tool call gets a result and the conversation stays valid."""
        fn = call.get("function", {})
        name = fn.get("name", "")
        start = time.time()
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            args, error = None, f"Error: the arguments aren't valid JSON ({e})."
        self.log.event("tool_start", tool=name, tool_use_id=call.get("id"), input=_truncate(args if args is not None else fn.get("arguments"), 2000))
        failed = False
        if args is None:
            result, failed = error, True
        else:
            try:
                handler = getattr(self, f"tool_{name}", None)
                if handler is None:
                    raise ToolError(f"There is no tool called {name!r}.")
                result = await self.cancellable(handler(args))
                if result is None:
                    result = "Interrupted: the run's time or cost limit was reached."
            except ToolError as e:
                result, failed = f"Error: {e}", True
            except Exception as e:  # a harness bug shouldn't break the conversation
                result, failed = f"Error: {type(e).__name__}: {e}", True
        duration = time.time() - start
        stats = self.tool_stats.setdefault(name, {"calls": 0, "seconds": 0.0, "failures": 0})
        stats["calls"] += 1
        stats["seconds"] += duration
        stats["failures"] += int(failed)
        self.log.event("tool_end", tool=name, tool_use_id=call.get("id"), duration_s=round(duration, 3), failed=failed,
                       response_chars=len(result), response=_truncate(result, 2000))
        self.last_message_at = time.time()
        return result

    def resolve(self, path: str, write: bool) -> Path:
        p = Path(path)
        real = (p if p.is_absolute() else self.workspace / p).resolve()
        roots = [self.workspace.resolve()] if write else self.read_roots
        if not any(real.is_relative_to(root) for root in roots):
            where = "inside the workspace" if write else "inside the workspace or the bess/harness sources"
            raise ToolError(f"{path} is outside the allowed paths: {'writes' if write else 'reads'} must be {where}.")
        return real

    def end_agent(self) -> None:
        if self.sandbox is not None:  # stops any background processes the agent left running
            self.sandbox.close()

    async def tool_bash(self, args: dict) -> str:
        command = args.get("command")
        if not command:
            raise ToolError("`command` is required.")
        timeout = min(float(args.get("timeout_s") or BASH_DEFAULT_S), BASH_MAX_S)
        if self.sandbox is None:
            tmp_dir = self.dir / "agent_tmp"  # the sandbox's /tmp
            tmp_dir.mkdir(exist_ok=True)
            self.sandbox = PersistentSandbox(self.workspace, tmp_dir)
        proc = await asyncio.create_subprocess_exec(
            *self.sandbox.command(command),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        chunks, size = [], 0

        async def pump():
            nonlocal size
            while data := await proc.stdout.read(65536):
                if size < 10 * OUTPUT_CHARS * 4:  # keep enough for the head and tail
                    chunks.append(data)
                size += len(data)

        reader = asyncio.create_task(pump())
        note = None
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            note = f"timed out after {timeout:.0f} s and was killed"
        except asyncio.CancelledError:
            self._kill(proc)
            reader.cancel()
            raise
        if note:
            self._kill(proc)
            await proc.wait()
        # A background process the command started may still hold its output
        # open; don't wait for it.
        try:
            await asyncio.wait_for(reader, 2)
        except asyncio.TimeoutError:
            reader.cancel()
        out = b"".join(chunks).decode(errors="replace")
        status = note or f"exit code {proc.returncode}"
        return f"{_clip(out)}\n[{status}]" if out else f"[{status}, no output]"

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def tool_read_file(self, args: dict) -> str:
        path = self.resolve(args.get("path", ""), write=False)
        if not path.is_file():
            raise ToolError(f"{args.get('path')} is not a file.")
        lines = path.read_text(errors="replace").splitlines()
        start = max(int(args.get("offset") or 1), 1)
        limit = int(args.get("limit") or READ_LIMIT_LINES)
        shown = [f"{n:6}\t{line[:2000]}" for n, line in enumerate(lines[start - 1 : start - 1 + limit], start)]
        more = len(lines) - (start - 1 + len(shown))
        return _clip("\n".join(shown)) + (f"\n[{more} more lines]" if more > 0 else "")

    async def tool_write_file(self, args: dict) -> str:
        path = self.resolve(args.get("path", ""), write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.get("content", ""))
        return f"Wrote {path.relative_to(self.workspace.resolve())} ({len(args.get('content', ''))} characters)."

    async def tool_edit_file(self, args: dict) -> str:
        path = self.resolve(args.get("path", ""), write=True)
        if not path.is_file():
            raise ToolError(f"{args.get('path')} is not a file.")
        text, old, new = path.read_text(), args.get("old_string", ""), args.get("new_string", "")
        count = text.count(old) if old else 0
        if count == 0:
            raise ToolError("`old_string` wasn't found in the file.")
        if count > 1 and not args.get("replace_all"):
            raise ToolError(f"`old_string` occurs {count} times; make it unique or set replace_all.")
        path.write_text(text.replace(old, new))
        return f"Replaced {count if args.get('replace_all') else 1} occurrence(s) in {path.relative_to(self.workspace.resolve())}."

    async def tool_submit_policy(self, args: dict) -> str:
        return await self.submit(args.get("summary", ""), args.get("rationale", ""))

    async def tool_status(self, args: dict) -> str:
        return self.status_text()

    async def tool_finish(self, args: dict) -> str:
        return self.finish(args.get("summary", ""))
