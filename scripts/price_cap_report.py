"""Render scripts/price_cap_analysis.py's results as one HTML page.

    uv run python scripts/price_cap_report.py   # -> data/analysis/price_cap_report.html
"""

import json
import math
import sys
from html import escape
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.normalise import PRICE_MAP, mpc_at  # noqa: E402
from harness.splits import DEV_END, EXCLUDED, REGIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
RAW_DB = ROOT / "data" / "benchmark.duckdb"
CHOSEN = "knee 1000, gamma 1"  # the map the data is built with (bess.normalise.PRICE_MAP)

MAPS = [  # (name in the summary, label, series colour token)
    ("raw", "Raw prices", "--s1"),
    ("constant scaling", "Constant scaling", "--s2"),
    ("knee 300, gamma 1", "Linear tail from $300", "--s3"),
    ("knee 1000, gamma 1", "Linear tail from $1,000 (chosen)", "--s4"),
]


def fy_label(fy: str) -> str:
    y = int(fy[2:])
    return f"{y - 1}–{str(y)[2:]}"


def tail_chart(tail: dict) -> str:
    """Log-scale lines: the cap and the quantiles of prices above $300, by year."""
    fys = list(tail)
    W, H, L, R, T, B = 640, 300, 64, 150, 16, 36
    lo, hi = math.log10(300), math.log10(30000)
    x = lambda i: L + i * (W - L - R) / (len(fys) - 1)  # noqa: E731
    y = lambda v: T + (hi - math.log10(v)) / (hi - lo) * (H - T - B)  # noqa: E731
    series = [
        ("Price cap", [tail[f]["mpc"] for f in fys], "--cap", "6 3"),
        ("99th percentile", [tail[f]["abs_quantiles"]["0.99"] for f in fys], "--s2", ""),
        ("90th percentile", [tail[f]["abs_quantiles"]["0.9"] for f in fys], "--s3", ""),
        ("Median", [tail[f]["abs_quantiles"]["0.5"] for f in fys], "--s4", ""),
    ]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Price cap and spike price percentiles by year">']
    for v in (300, 1000, 3000, 10000, 30000):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">${v:,}</text>')
    for i, f in enumerate(fys):
        out.append(f'<text x="{x(i):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{fy_label(f)}</text>')
    for name, vals, color, dash in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="var({color})" stroke-width="2"'
                   f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="var({color})" stroke="var(--bg)" '
                       f'stroke-width="2"><title>{name}, {fy_label(fys[i])}: ${v:,.0f}</title></circle>')
        out.append(f'<text x="{W - R + 10}" y="{y(vals[-1]) + 4:.1f}" class="label">{name}</text>')
    out.append("</svg>")
    return "".join(out)


def arbitrage_chart(maps: dict) -> str:
    """Grouped bars: mean daily arbitrage value per year under each map (dev)."""
    by = {name: {r["fy"]: r["arbitrage"] for r in maps[name]["by_fy"]} for name, _, _ in MAPS}
    fys = list(by["raw"])
    W, H, L, R, T, B = 640, 300, 56, 8, 16, 36
    top = 1200
    y = lambda v: T + (1 - v / top) * (H - T - B)  # noqa: E731
    group = (W - L - R) / len(fys)
    bw = (group - 18) / len(MAPS)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean daily arbitrage value by year under each map">']
    for v in range(0, top + 1, 300):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">${v:,}</text>')
    for i, f in enumerate(fys):
        x0 = L + i * group + 9
        label = fy_label(f) + (" *" if f == fys[-1] else "")
        out.append(f'<text x="{x0 + group / 2 - 9:.1f}" y="{H - 12}" class="tick" text-anchor="middle">{label}</text>')
        for j, (name, lab, color) in enumerate(MAPS):
            v = by[name][f]
            h = max(y(0) - y(v), 0)
            out.append(f'<rect x="{x0 + j * bw + 1:.1f}" y="{y(v):.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="2" '
                       f'fill="var({color})"><title>{lab}, {fy_label(f)}: ${v:,.0f} a day</title></rect>')
    out.append("</svg>")
    return "".join(out)


def scatter_data() -> dict:
    """Per region-day: arbitrage value (from the analysis, raw and under the
    chosen map) against the mean price, the $300 cap payout and the payout
    above $1,000, each from the same prices. Compact arrays for the page."""
    con = duckdb.connect(str(RAW_DB), read_only=True)
    try:
        prices = con.execute("""SELECT SETTLEMENTDATE, REGIONID, RRP FROM dispatchprice
                                WHERE INTERVENTION = 0 ORDER BY REGIONID, SETTLEMENTDATE""").df()
        thresholds = con.execute("SELECT * FROM market_price_thresholds").df()
    finally:
        con.close()
    start = prices.SETTLEMENTDATE - pd.Timedelta(minutes=5)
    prices["day"] = (start - pd.Timedelta(hours=4)).dt.normalize()
    for x0, x1 in EXCLUDED[:1]:  # the analysis left out only the suspension
        prices = prices[~((prices.SETTLEMENTDATE > x0) & (prices.SETTLEMENTDATE <= x1))]
    prices = prices[prices.day >= pd.Timestamp("2021-10-01")]
    mapped = PRICE_MAP(prices.RRP.to_numpy(), mpc_at(thresholds, prices.SETTLEMENTDATE - pd.Timedelta(minutes=5)))
    daily = pd.read_parquet(OUT / "price_cap_daily.parquet", columns=["region", "day", "arbitrage", "n", "map"])
    out = {"regions": REGIONS, "dev_end": str(DEV_END.date())}
    for key, name, p in [("raw", "raw", prices.RRP.to_numpy()), ("mapped", CHOSEN, mapped)]:
        f = pd.DataFrame({"region": prices.REGIONID.to_numpy(), "day": prices.day.to_numpy(), "p": p})
        g = f.groupby(["region", "day"]).p
        stats = pd.DataFrame({
            "swap": g.mean(),
            "cap300": g.apply(lambda v: np.maximum(v - 300, 0).mean()),
            "cap1000": g.apply(lambda v: np.maximum(v - 1000, 0).mean()),
        }).reset_index()
        d = daily[(daily["map"] == name) & (daily.n == 288)].merge(stats, on=["region", "day"])
        d = d.sort_values(["day", "region"])
        fy = d.day.dt.year + (d.day.dt.month >= 7)
        out[key] = {
            "arb": d.arbitrage.round(1).tolist(), "swap": d.swap.round(2).tolist(),
            "cap300": d.cap300.round(3).tolist(), "cap1000": d.cap1000.round(3).tolist(),
            "fy": fy.astype(int).tolist(), "region": d.region.map(REGIONS.index).tolist(),
            "day": d.day.dt.strftime("%Y-%m-%d").tolist(),
        }
    return out


# The scatter's drawing code (canvas, hover, filters), outside the f-string so
# its braces stay single.
SCATTER_JS = r"""<script>
const SC = __SCATTER__;
(function () {
  const ySel = document.getElementById("sc-y"), yearSel = document.getElementById("sc-year");
  const years = [...new Set(SC.raw.fy)].sort();
  const fyLabel = y => `${y - 1}–${String(y).slice(2)}` + (y >= 2026 ? " (holdout)" : "");
  yearSel.innerHTML = `<option value="all">All years</option>` + years.map(y => `<option value="${y}">${fyLabel(y)}</option>`).join("");
  const sym = (v, c) => Math.sign(v) * Math.log10(1 + Math.abs(v) / c);
  const C = { arb: 10, swap: 10, cap300: 0.1, cap1000: 0.1 };
  const color = i => getComputedStyle(document.documentElement).getPropertyValue(`--k${i + 1}`).trim();
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const panels = { raw: document.getElementById("sc-raw"), mapped: document.getElementById("sc-mapped") };
  let drawn = {};
  function niceTicks(lo, hi, c) {
    const out = [0];
    for (const s of [1, -1]) for (let e = -2; e <= 6; e++) for (const m of [1, 10]) {
      const v = s * m * 10 ** e; if (m === 10) continue;
      const t = sym(v, c); if (t > lo && t < hi && Math.abs(v) >= c) out.push(v);
    }
    return [...new Set(out)].filter(v => sym(v, c) >= lo && sym(v, c) <= hi);
  }
  const fmt = v => { const a = Math.abs(v); return (v < 0 ? "−" : "") + "$" + (a >= 1000 ? (a / 1000) + "k" : a >= 1 ? a.toFixed(0) : a.toString()); };
  function draw() {
    const yk = ySel.value, yr = yearSel.value;
    const sel = d => d.fy.map((f, i) => (yr === "all" || f === +yr) ? i : -1).filter(i => i >= 0);
    const idx = { raw: sel(SC.raw), mapped: sel(SC.mapped) };
    let xs = [], ys = [];
    for (const k of ["raw", "mapped"]) for (const i of idx[k]) { xs.push(sym(SC[k].arb[i], C.arb)); ys.push(sym(SC[k][yk][i], C[yk])); }
    const pad = (a, b) => [a - (b - a) * 0.04, b + (b - a) * 0.04];
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys, 0), Math.max(...ys));
    const byYear = yr === "all";
    const groups = byYear ? years : SC.regions.map((_, i) => i);
    document.getElementById("sc-legend").innerHTML = '<ul class="legend">' + groups.map((g, i) =>
      `<li><span class="swatch" style="background:var(--k${i + 1})"></span>${byYear ? fyLabel(g) : SC.regions[g]}</li>`).join("") + "</ul>";
    document.getElementById("sc-note").textContent = `${idx.raw.length.toLocaleString()} region-days. Colour: ${byYear ? "financial year" : "region"}. Hover a point for its values.`;
    for (const k of ["raw", "mapped"]) {
      const cv = panels[k], dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const L = 52, R = 10, T = 10, B = 34;
      const X = v => L + (sym(v, C.arb) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - sym(v, C[yk])) / (y1 - y0) * (H - T - B);
      g.font = "11px " + css("--mono"); g.fillStyle = css("--muted"); g.strokeStyle = css("--rule"); g.lineWidth = 1;
      for (const v of niceTicks(x0, x1, C.arb)) { const x = X(v); g.beginPath(); g.moveTo(x, T); g.lineTo(x, H - B); g.stroke(); g.textAlign = "center"; g.fillText(fmt(v), x, H - B + 14); }
      for (const v of niceTicks(y0, y1, C[yk])) { const y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.textAlign = "right"; g.fillText(fmt(v), L - 6, y + 4); }
      g.textAlign = "center"; g.fillText("arbitrage value, $/day", L + (W - L - R) / 2, H - 4);
      const d = SC[k], pts = [];
      g.globalAlpha = byYear ? 0.45 : 0.6;
      for (const i of idx[k]) {
        const gi = byYear ? years.indexOf(d.fy[i]) : d.region[i];
        const x = X(d.arb[i]), y = Y(d[yk][i]);
        g.fillStyle = color(gi); g.beginPath(); g.arc(x, y, 2.2, 0, 2 * Math.PI); g.fill();
        pts.push([x, y, i]);
      }
      g.globalAlpha = 1;
      drawn[k] = { pts, yk };
    }
  }
  function hover(k, ev) {
    const cv = panels[k], tip = cv.parentElement.querySelector(".tip") || cv.parentElement.appendChild(Object.assign(document.createElement("div"), { className: "tip" }));
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 64;
    for (const [x, y, i] of drawn[k].pts) { const dd = (x - mx) ** 2 + (y - my) ** 2; if (dd < bd) { bd = dd; best = [x, y, i]; } }
    if (!best) { tip.hidden = true; return; }
    const d = SC[k], i = best[2], yk = drawn[k].yk;
    const names = { swap: "mean price", cap300: "payout above $300", cap1000: "payout above $1,000" };
    tip.innerHTML = `${SC.regions[d.region[i]]} ${d.day[i]}<br>arbitrage $${d.arb[i].toLocaleString()}/day<br>${names[yk]} $${d[yk][i].toLocaleString()}/MWh`;
    tip.hidden = false;
    tip.style.left = Math.min(best[0] + 10, cv.clientWidth - 190) + "px"; tip.style.top = (best[1] + 34) + "px";
  }
  for (const k of ["raw", "mapped"]) {
    panels[k].addEventListener("mousemove", e => hover(k, e));
    panels[k].addEventListener("mouseleave", () => { const t = panels[k].parentElement.querySelector(".tip"); if (t) t.hidden = true; });
  }
  ySel.addEventListener("change", draw); yearSel.addEventListener("change", draw);
  window.addEventListener("resize", draw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  draw();
})();
</script>
"""


def legend(items) -> str:
    return '<ul class="legend">' + "".join(
        f'<li><span class="swatch" style="background:var({c})"></span>{escape(lab)}</li>' for lab, c in items) + "</ul>"


def main() -> None:
    s = json.loads((OUT / "price_cap_summary.json").read_text())
    tail = s["tail_by_fy"]
    raw = {r["fy"]: r for r in s["raw_by_fy"]}
    maps = s["maps"]
    effect_rows = "".join(
        f'<tr><td>{escape(lab)}</td><td class="num">{maps[name]["arbitrage_change_pct"]:+.1f}%</td>'
        f'<td class="num">{maps[name]["swap_change_pct"]:+.1f}%</td>'
        f'<td class="num">{maps[name]["by_fy"][0]["spike_share"] * 100:.0f}%</td></tr>'
        for name, lab in [("raw", "Raw prices"), ("constant scaling", "Constant scaling (prices above $0 × cap ratio)"),
                          ("knee 300, gamma 0.8", "Tail from $300, γ = 0.8"), ("knee 300, gamma 1", "Tail from $300, linear"),
                          ("knee 300, gamma 1.25", "Tail from $300, γ = 1.25"), ("knee 300, gamma 1.5", "Tail from $300, γ = 1.5"),
                          ("knee 1000, gamma 1", "Tail from $1,000, linear (chosen)"),
                          ("knee 1000, gamma 1.5", "Tail from $1,000, γ = 1.5")])
    year_rows = "".join(
        f'<tr><td>{fy_label(fy)}</td><td class="num">${s["caps"][fy]["max"]:,.0f}</td><td class="num">{r["days"]:,}</td>'
        f'<td class="num">${r["arbitrage"]:,.0f}</td><td class="num">${r["swap"]:,.0f}</td>'
        f'<td class="num">${r["cap_payout"]:,.2f}</td><td class="num">{r["spike_share"] * 100:.0f}%</td>'
        f'<td class="num">{s["regression_raw_by_fy"][fy]["per_cap_payout"]:.1f}</td></tr>'
        for fy, r in raw.items())
    html = f"""<title>Price Cap Mapping</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; charts and tables at full column width, findings first. */
:root {{
  --bg: #f6f7f5; --panel: #ffffff; --ink: #1b2422; --muted: #5b6764; --rule: #d9dedb;
  --accent: #0f6e63; --cap: #b4561f;
  --s1: #8a9491; --s2: #b4561f; --s3: #c99a2e; --s4: #0f6e63;
  --k1: #2a6fb0; --k2: #c2410c; --k3: #0f8a6e; --k4: #9d3ea8; --k5: #a37a00; --k6: #5b6764;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0;
  --k1: #6fa8e0; --k2: #f08a50; --k3: #47c4a2; --k4: #d07fdb; --k5: #e0bb45; --k6: #a3b0ac; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0;
  --k1: #6fa8e0; --k2: #f08a50; --k3: #47c4a2; --k4: #d07fdb; --k5: #e0bb45; --k6: #a3b0ac; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }}
main {{ max-width: 46rem; margin: 0 auto; padding-inline: 16px; padding-block: 40px 64px; }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; line-height: 1.15; }}
h1 {{ font-size: 2.1rem; margin: 0 0 .4rem; }}
h2 {{ font-size: 1.3rem; margin: 2.6rem 0 .6rem; }}
.eyebrow {{ font: 500 .75rem var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
p {{ max-width: 65ch; margin: .6rem 0; }}
.lede {{ font-size: 1.05rem; color: var(--muted); }}
.decision {{ background: var(--panel); border: 1px solid var(--rule); border-left: 4px solid var(--cap);
  border-radius: 4px; padding: 14px 18px; margin: 1.6rem 0; }}
.decision strong {{ font-family: var(--display); font-size: 1.05rem; }}
figure {{ margin: 1rem 0; }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 11px var(--mono); }}
svg .label {{ fill: var(--ink); font: 12px var(--body); }}
figcaption {{ color: var(--muted); font-size: .85rem; }}
.legend {{ list-style: none; padding: 0; margin: .4rem 0; display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .85rem; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: left; }}
th {{ font: 500 .72rem var(--mono); text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }}
td.num, th.num {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
code {{ font-family: var(--mono); font-size: .9em; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; margin: .8rem 0; font-size: .88rem; }}
.controls label {{ display: flex; gap: 8px; align-items: center; }}
select {{ font: inherit; padding: 4px 8px; border: 1px solid var(--rule); border-radius: 4px; background: var(--panel); color: var(--ink); }}
select:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 14px; }}
.panel {{ position: relative; min-width: 0; }}
.panel h3 {{ font: 600 .95rem var(--display); margin: 0 0 4px; }}
.panel canvas {{ width: 100%; aspect-ratio: 1 / 0.85; max-width: 100%; display: block; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 4px; }}
.tip {{ position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: 6px 8px; font: 12px/1.4 var(--mono); box-shadow: 0 2px 8px rgb(0 0 0 / .15); white-space: nowrap; z-index: 2; }}
.c1 {{ --c: var(--k1); }}
</style>
<main>
<div class="eyebrow">Storage optimisation benchmark · data preparation</div>
<h1>Price Cap Mapping</h1>
<p class="lede">The NEM's market price cap rose from $15,100/MWh in October 2021 to $23,200/MWh in July 2026. Before
five years of prices can be scored together, older prices need mapping onto today's cap. This is the evidence for how.</p>

<div class="decision"><strong>Decision for you: the knee.</strong>
<p>The data is now built with prices up to <b>$1,000/MWh unchanged</b> and a linear stretch above it that lands each
year's cap on $23,200. That changes the battery's mean dev arbitrage value by +16%. The plan's starting point was a
$300 knee (+20%). Constant scaling would change it by +40%. Switching is one constant in <code>bess/normalise.py</code>
and an 8-second rebuild; the reference scores and the static-bid fit would need re-running.</p></div>

<h2>Spike prices don't follow the cap; only the top of the tail does</h2>
<p>For every 5-minute price above $300 in the dev period, the median and 90th percentile stay near $360–390 and
$570–1,000 as the cap rises by a third. The 99th percentile and the maximum sit at or near the cap. So prices in the
hundreds come from bids, and only prices near the cap are set by it. A map that stretches the whole tail inflates
those ordinary prices; a knee at about $1,000 leaves them alone.</p>
<figure>{tail_chart(tail)}
<figcaption>Prices above $300/MWh, all regions, dev period (October 2021 to August 2025; the last year is July–August
2025 only, under the $20,300 cap). Log scale. Hover a point for its value.</figcaption></figure>

<h2>What each map does to a battery's value</h2>
<p>For each region and trading day, the benchmark battery's perfect-foresight arbitrage value (1 MW / 2 MWh, starting
and ending at 50%) was solved again on the mapped prices. Spikes earn most of a battery's value, so every map that pins
the cap raises it; the question is how much of that is the cap and how much is inflated ordinary spikes.</p>
<figure>{legend([(lab, c) for _, lab, c in MAPS])}{arbitrage_chart(maps)}
<figcaption>Mean daily arbitrage value per region-day, dev period. * July–August 2025 only.</figcaption></figure>
<div class="table-wrap"><table>
<thead><tr><th>Map</th><th class="num">Arbitrage</th><th class="num">Mean price</th><th class="num">Spike share, 2021–22</th></tr></thead>
<tbody>{effect_rows}</tbody></table></div>
<p>Changes are over the whole dev period against raw prices. A γ above 1 bends the tail below the identity, so prices
just above the knee come out <i>lower</i> than they were; that's why those rows change value least, and why they're
not candidates.</p>

<h2>Day by day: arbitrage against what the day's prices pay</h2>
<p>Each point is one region-day: the battery's perfect-foresight arbitrage value (x) against a measure of that
day's prices (y), on the raw prices and on the mapped ones. A good map moves older days toward the recent pattern
without bending the relationship. Below $1,000 the chosen map changes nothing, so the payout above $1,000 is the
measure that shows the stretch most directly; the mean price and the $300 payout mix stretched and unchanged
prices. Both axes are symmetric-log. 2025–26 and 2026–27 are holdout years, shown for context.</p>
<div class="controls">
  <label for="sc-y">Y axis <select id="sc-y">
    <option value="cap1000">Payout above $1,000/MWh ($/MWh, daily mean)</option>
    <option value="cap300">Payout above $300/MWh ($/MWh, daily mean)</option>
    <option value="swap">Mean price ($/MWh)</option></select></label>
  <label for="sc-year">Year <select id="sc-year"></select></label>
</div>
<div id="sc-legend"></div>
<div class="panels">
  <div class="panel"><h3>Raw prices</h3><canvas id="sc-raw" aria-label="Scatter of daily arbitrage against the chosen price measure, raw prices"></canvas></div>
  <div class="panel"><h3>Mapped onto $23,200 (knee $1,000)</h3><canvas id="sc-mapped" aria-label="The same scatter on mapped prices"></canvas></div>
</div>
<p class="note" id="sc-note" style="color:var(--muted);font-size:.85rem"></p>

<h2>Year by year, raw prices</h2>
<div class="table-wrap"><table>
<thead><tr><th>Year</th><th class="num">Cap</th><th class="num">Region-days</th><th class="num">Arbitrage /day</th>
<th class="num">Mean price</th><th class="num">$300 cap payout</th><th class="num">From spikes</th>
<th class="num">$ per $ of cap payout</th></tr></thead>
<tbody>{year_rows}</tbody></table></div>
<p>Financial years (July to June). 2025–26 and 2026–27 are the holdout and are shown for context only; nothing here
was fitted to them. "From spikes" is the share of arbitrage revenue earned in intervals above $300. The last column is
the slope of daily arbitrage on the daily cap payout (with the mean price as a second regressor), which falls from
about 23 to 15 as the market changes; no map fixes that, since it isn't about the cap.</p>

<h2>How it was done</h2>
<p><code>scripts/price_cap_analysis.py</code> solves about 8,750 daily MILPs per map (5 regions, October 2021 to August
2026, excluding the June 2022 suspension), with a 0.1% optimality gap. The cap in force comes from AEMO's
<code>MARKET_PRICE_THRESHOLDS</code>. Forecast prices are mapped with the cap in force when the forecast ran. The
market floor (−$1,000) hasn't changed, so negative prices are never mapped.</p>
</main>
"""
    html += SCATTER_JS.replace("__SCATTER__", json.dumps(scatter_data(), separators=(",", ":")))
    (OUT / "price_cap_report.html").write_text(html)
    print(OUT / "price_cap_report.html")


if __name__ == "__main__":
    main()
