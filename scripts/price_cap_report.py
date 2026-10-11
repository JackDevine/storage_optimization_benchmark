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
    """Per region-week (ISO weeks, Monday to Sunday, by trading day): mean daily
    arbitrage value (from the analysis, raw and under the chosen map, and
    constant scaling) against the week's mean price, $300 cap payout and
    payout above $1,000, each from the same prices. Compact arrays for the page."""
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
    mpc = mpc_at(thresholds, prices.SETTLEMENTDATE - pd.Timedelta(minutes=5))
    mapped = PRICE_MAP(prices.RRP.to_numpy(), mpc)
    # For comparison: every positive price scaled by MPC_now / MPC_then (as in
    # the analysis's "constant scaling"; the floor hasn't changed, so negative
    # prices aren't scaled).
    raw = prices.RRP.to_numpy()
    scaled = np.where(raw > 0, raw * PRICE_MAP.mpc_now / mpc, raw)
    daily = pd.read_parquet(OUT / "price_cap_daily.parquet", columns=["region", "day", "arbitrage", "n", "map"])
    out = {"regions": REGIONS, "dev_end": str(DEV_END.date())}
    for key, name, p in [("raw", "raw", raw), ("mapped", CHOSEN, mapped), ("scaled", "constant scaling", scaled)]:
        f = pd.DataFrame({"region": prices.REGIONID.to_numpy(), "day": prices.day.to_numpy(), "p": p})
        g = f.groupby(["region", "day"]).p
        stats = pd.DataFrame({
            "swap": g.mean(),
            "cap300": g.apply(lambda v: np.maximum(v - 300, 0).mean()),
            "cap1000": g.apply(lambda v: np.maximum(v - 1000, 0).mean()),
        }).reset_index()
        d = daily[(daily["map"] == name) & (daily.n == 288)].merge(stats, on=["region", "day"])
        d["week"] = d.day.dt.to_period("W-SUN").dt.start_time  # the Monday starting the trading day's ISO week
        d = (d.groupby(["region", "week"])[["arbitrage", "swap", "cap300", "cap1000"]].mean()
             .reset_index().rename(columns={"week": "day"}).sort_values(["day", "region"]))
        fy = d.day.dt.year + (d.day.dt.month >= 7)
        out[key] = {
            "arb": d.arbitrage.round(1).tolist(), "swap": d.swap.round(2).tolist(),
            "cap300": d.cap300.round(3).tolist(), "cap1000": d.cap1000.round(3).tolist(),
            "fy": fy.astype(int).tolist(), "region": d.region.map(REGIONS.index).tolist(),
            "day": d.day.dt.strftime("week of %d %b %Y").tolist(),
            "start": d.day.dt.strftime("%Y-%m-%d").tolist(),
        }
    return out


# The scatter's drawing code (canvas, hover, filters), outside the f-string so
# its braces stay single.
SCATTER_JS = r"""<script>
const SC = __SCATTER__;
(function () {
  const ySel = document.getElementById("sc-y"), yearSel = document.getElementById("sc-year");
  const regionSel = document.getElementById("sc-region");
  regionSel.innerHTML = `<option value="all">All regions</option>` + SC.regions.map((r, i) => `<option value="${i}">${r}</option>`).join("");
  const years = [...new Set(SC.raw.fy)].sort();
  const fyLabel = y => `${y - 1}–${String(y).slice(2)}` + (y >= 2026 ? " (holdout)" : "");
  yearSel.innerHTML = `<option value="all">All years</option>` + years.map(y => `<option value="${y}">${fyLabel(y)}</option>`).join("");
  const sym = (v, c) => Math.sign(v) * Math.log10(1 + Math.abs(v) / c);
  const C = { arb: 10, swap: 10, cap300: 0.1, cap1000: 0.1 };
  const color = i => getComputedStyle(document.documentElement).getPropertyValue(`--k${i + 1}`).trim();
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const KEYS = ["raw", "mapped", "scaled"];
  const panels = Object.fromEntries(KEYS.map(k => [k, document.getElementById("sc-" + k)]));
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
  const MAPS = [["raw", "Raw prices", "--s1", "circle"], ["mapped", "$1,000 knee (chosen)", "--s4", "square"],
                ["scaled", "Constant scaling", "--s2", "triangle"]];
  const comb = document.getElementById("sc-combined"), links = document.getElementById("sc-links");
  document.getElementById("sc-combined-legend").innerHTML = MAPS.map(([, lab, c, shape]) =>
    `<li><svg class="marker" viewBox="0 0 12 12">${shape === "circle" ? '<circle cx="6" cy="6" r="5"' :
      shape === "square" ? '<rect x="1.5" y="1.5" width="9" height="9"' : '<path d="M6 1 L11 11 L1 11 Z"'} fill="var(${c})"/></svg>${lab}</li>`).join("");
  function marker(g, shape, x, y, r) {
    g.beginPath();
    if (shape === "circle") g.arc(x, y, r, 0, 2 * Math.PI);
    else if (shape === "square") g.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7);
    else { g.moveTo(x, y - r * 1.1); g.lineTo(x + r, y + r * 0.8); g.lineTo(x - r, y + r * 0.8); g.closePath(); }
    g.fill();
  }
  function frame(cv, x0, x1, y0, y1, yk) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const L = 52, R = 10, T = 10, B = 34;
    const X = v => L + (sym(v, C.arb) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - sym(v, C[yk])) / (y1 - y0) * (H - T - B);
    g.font = "11px " + css("--mono"); g.fillStyle = css("--muted"); g.strokeStyle = css("--rule"); g.lineWidth = 1;
    for (const v of niceTicks(x0, x1, C.arb)) { const x = X(v); g.beginPath(); g.moveTo(x, T); g.lineTo(x, H - B); g.stroke(); g.textAlign = "center"; g.fillText(fmt(v), x, H - B + 14); }
    for (const v of niceTicks(y0, y1, C[yk])) { const y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.textAlign = "right"; g.fillText(fmt(v), L - 6, y + 4); }
    g.textAlign = "center"; g.fillText("arbitrage value, $/day", L + (W - L - R) / 2, H - 4);
    return { g, X, Y };
  }
  function draw() {
    const yk = ySel.value, yr = yearSel.value, rg = regionSel.value;
    const sel = d => d.fy.map((f, i) => ((yr === "all" || f === +yr) && (rg === "all" || d.region[i] === +rg)) ? i : -1)
      .filter(i => i >= 0);
    const idx = Object.fromEntries(KEYS.map(k => [k, sel(SC[k])]));
    let xs = [], ys = [];
    for (const k of KEYS) for (const i of idx[k]) { xs.push(sym(SC[k].arb[i], C.arb)); ys.push(sym(SC[k][yk][i], C[yk])); }
    const pad = (a, b) => [a - (b - a) * 0.04, b + (b - a) * 0.04];
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys, 0), Math.max(...ys));
    const byYear = yr === "all" || rg !== "all";  // one region: colour by year
    const groups = byYear ? years : SC.regions.map((_, i) => i);
    document.getElementById("sc-legend").innerHTML = '<ul class="legend">' + groups.map((g, i) =>
      `<li><span class="swatch" style="background:var(--k${i + 1})"></span>${byYear ? fyLabel(g) : SC.regions[g]}</li>`).join("") + "</ul>";
    document.getElementById("sc-note").textContent = `${idx.raw.length.toLocaleString()} region-weeks. Colour: ${byYear ? "financial year" : "region"}. Hover a point for its values.`;
    {
      const { g, X, Y } = frame(comb, x0, x1, y0, y1, yk), pts = [];
      if (links.checked) {
        g.strokeStyle = css("--muted"); g.globalAlpha = 0.25; g.lineWidth = 1;
        for (const i of idx.raw) {
          g.beginPath(); g.moveTo(X(SC.raw.arb[i]), Y(SC.raw[yk][i]));
          g.lineTo(X(SC.mapped.arb[i]), Y(SC.mapped[yk][i])); g.lineTo(X(SC.scaled.arb[i]), Y(SC.scaled[yk][i])); g.stroke();
        }
      }
      g.globalAlpha = 0.7;
      for (const [k, , c, shape] of MAPS) {
        g.fillStyle = css(c);
        for (const i of idx[k]) { const x = X(SC[k].arb[i]), y = Y(SC[k][yk][i]); marker(g, shape, x, y, 3); pts.push([x, y, i, k]); }
      }
      g.globalAlpha = 1;
      drawn.combined = { pts, yk };
    }
    for (const k of KEYS) {
      const { g, X, Y } = frame(panels[k], x0, x1, y0, y1, yk);
      const d = SC[k], pts = [];
      g.globalAlpha = 0.65;
      for (const i of idx[k]) {
        const gi = byYear ? years.indexOf(d.fy[i]) : d.region[i];
        const x = X(d.arb[i]), y = Y(d[yk][i]);
        g.fillStyle = color(gi); g.beginPath(); g.arc(x, y, 2.8, 0, 2 * Math.PI); g.fill();
        pts.push([x, y, i, k]);
      }
      g.globalAlpha = 1;
      drawn[k] = { pts, yk };
    }
  }
  function hover(k, ev) {
    const cv = k === "combined" ? comb : panels[k], tip = cv.parentElement.querySelector(".tip") || cv.parentElement.appendChild(Object.assign(document.createElement("div"), { className: "tip" }));
    const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 64;
    for (const [x, y, i, mk] of drawn[k].pts) { const dd = (x - mx) ** 2 + (y - my) ** 2; if (dd < bd) { bd = dd; best = [x, y, i, mk]; } }
    if (!best) { tip.hidden = true; return; }
    const d = SC[best[3]], i = best[2], yk = drawn[k].yk;
    const names = { swap: "mean price", cap300: "payout above $300", cap1000: "payout above $1,000" };
    const mapName = k === "combined" ? MAPS.find(m => m[0] === best[3])[1] + "<br>" : "";
    tip.innerHTML = `${mapName}${SC.regions[d.region[i]]} ${d.day[i]}<br>arbitrage $${d.arb[i].toLocaleString()}/day (mean)<br>${names[yk]} $${d[yk][i].toLocaleString()}/MWh`;
    tip.hidden = false;
    tip.style.left = Math.min(best[0] + 10, cv.clientWidth - 190) + "px"; tip.style.top = (best[1] + 34) + "px";
  }
  for (const k of [...KEYS, "combined"]) {
    const cv = k === "combined" ? comb : panels[k];
    cv.addEventListener("mousemove", e => hover(k, e));
    cv.addEventListener("mouseleave", () => { const t = cv.parentElement.querySelector(".tip"); if (t) t.hidden = true; });
  }
  links.addEventListener("change", draw);
  ySel.addEventListener("change", draw); yearSel.addEventListener("change", draw);
  regionSel.addEventListener("change", draw);
  window.addEventListener("resize", draw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  draw();

  // ---- correlation matrices by financial year ----
  const methodSel = document.getElementById("cm-method"), holdoutBox = document.getElementById("cm-holdout");
  const VARS = [["arb", "arbitrage"], ["swap", "mean price"], ["cap300", "> $300"], ["cap1000", "> $1,000"]];
  const devEnd = new Date(SC.dev_end);
  const isDev = d => i => (new Date(d.start[i]).getTime() + 7 * 864e5) <= devEnd.getTime();
  function ranks(a) {
    const o = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = new Array(a.length);
    for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++;
      for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1; }
    return r;
  }
  function corr(a, b) {
    const n = a.length; if (n < 3) return NaN;
    const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : NaN;
  }
  function matrix(d, rows, method) {
    let cols = VARS.map(([k]) => rows.map(i => d[k][i]));
    if (method === "spearman") cols = cols.map(ranks);
    return cols.map(a => cols.map(b => corr(a, b)));
  }
  function cell(v, self) {
    if (self) return '<td class="self">1</td>';
    if (!isFinite(v)) return "<td>–</td>";
    const pct = Math.round(Math.abs(v) * 70);
    return `<td style="background:color-mix(in srgb, var(${v >= 0 ? "--pos" : "--neg"}) ${pct}%, var(--panel))">${v.toFixed(2)}</td>`;
  }
  function drawMatrices() {
    const method = methodSel.value, rg = regionSel.value, withHoldout = holdoutBox.checked;
    const d0 = SC.raw;
    const base = d0.fy.map((_, i) => i).filter(i => (rg === "all" || d0.region[i] === +rg) && (withHoldout || isDev(d0)(i)));
    const yearsHere = [...new Set(base.map(i => d0.fy[i]))].sort();
    const blocks = [["all", base], ...yearsHere.map(y => [y, base.filter(i => d0.fy[i] === y)])];
    const names = { raw: "Raw prices", mapped: "$1,000 knee", scaled: "Constant scaling" };
    let html = "";
    for (const [y, rows] of blocks) {
      const above = rows.filter(i => d0.cap1000[i] > 0).length;
      const flags = [];
      if (rows.length < 60) flags.push(`small sample: ${rows.length} region-weeks`);
      if (above < 15) flags.push(`only ${above} region-weeks with any raw price above $1,000`);
      const partial = y === 2026 && !withHoldout ? " (July–August 2025 only)" : "";
      html += `<div class="cm-year">${y === "all" ? (withHoldout ? "All years" : "All dev years") : fyLabel(y) + partial}` +
        `<small>${rows.length} region-weeks, ${above} with raw prices above $1,000` +
        (flags.length ? ` · <span class="flag">${flags.join("; ")}</span>` : "") + "</small></div><div class=\"cm-row\">";
      for (const k of KEYS) {
        const m = matrix(SC[k], rows, method);
        html += `<figure><figcaption>${names[k]}</figcaption><table class="cm"><thead><tr><th></th>` +
          VARS.map(([, l]) => `<th>${l}</th>`).join("") + "</tr></thead><tbody>" +
          m.map((r, a) => `<tr><th>${VARS[a][1]}</th>` + r.map((v, b) => cell(v, a === b)).join("") + "</tr>").join("") +
          "</tbody></table></figure>";
      }
      html += "</div>";
    }
    document.getElementById("cm").innerHTML = html;
    document.getElementById("cm-note").textContent =
      `${method === "pearson" ? "Pearson" : "Spearman"} correlation, ${rg === "all" ? "all regions" : SC.regions[+rg]}. ` +
      "Blue is positive, orange negative. Weeks are dev weeks unless holdout years are included; a week counts as dev if it ends by " + SC.dev_end + ".";
  }
  methodSel.addEventListener("change", drawMatrices); holdoutBox.addEventListener("change", drawMatrices);
  regionSel.addEventListener("change", drawMatrices);
  drawMatrices();
})();
</script>
"""


# The covariance / correlation tables (added from the published page by another session).
COVARIANCE_JS = r"""
<script>
(function () {
  const VARS = [["swap", "mean_price"], ["cap300", "cap_payout"], ["cap1000", "value_above_1000"], ["arb", "arbitrage_payout"]];
  const MAPS = [["raw", "Raw prices"], ["mapped", "$1,000 knee mapped"], ["scaled", "Constant scaling ($0 knee)"]];
  const yearSel = document.getElementById("sc-year"), regionSel = document.getElementById("sc-region"), mode = document.getElementById("cv-mode"), per = document.getElementById("cv-period");
  const devEnd = new Date(SC.dev_end), isDev = i => new Date(SC.raw.day[i].replace("week of ", "")) < devEnd;
  const sig = v => { if (v === 0) return "0"; const a = Math.abs(v);
    return a >= 1000 ? Math.round(v).toLocaleString() : v.toPrecision(3).replace(/\.?0+$/, m => m.includes(".") ? "" : m); };
  function stats(d, idx) {
    const n = idx.length, cols = VARS.map(([k]) => idx.map(i => d[k][i]));
    const mu = cols.map(c => c.reduce((a, b) => a + b, 0) / n);
    const cov = cols.map((a, p) => cols.map((b, q) => a.reduce((s, x, j) => s + (x - mu[p]) * (b[j] - mu[q]), 0) / (n - 1)));
    const corr = cov.map((r, p) => r.map((v, q) => v / Math.sqrt(cov[p][p] * cov[q][q])));
    return { n, cov, corr };
  }
  function draw() {
    const yr = yearSel.value, rg = regionSel.value, m = mode.value;
    const idx = SC.raw.fy.map((f, i) => ((yr === "all" || f === +yr) && (rg === "all" || SC.raw.region[i] === +rg) && (per.value === "all" || isDev(i))) ? i : -1).filter(i => i >= 0);
    if (idx.length < 3) { document.getElementById("cv-tables").innerHTML = "<p>Too few region-weeks for this selection.</p>"; return; }
    const head = `<tr><th></th>${VARS.map(([, n]) => `<th class="num"><code>${n}</code></th>`).join("")}</tr>`;
    document.getElementById("cv-tables").innerHTML = MAPS.map(([k, lab]) => {
      const s = stats(SC[k], idx), M = s[m];
      const body = VARS.map(([, n], p) => `<tr><th><code>${n}</code></th>${M[p].map((v, q) =>
        `<td class="num${p === q ? " diag" : ""}">${isFinite(v) ? (m === "corr" ? v.toFixed(2) : sig(v)) : "–"}</td>`).join("")}</tr>`).join("");
      return `<div class="cv-block"><h4>${lab}</h4><div class="table-wrap"><table><thead>${head}</thead><tbody>${body}</tbody></table></div></div>`;
    }).join("");
    document.getElementById("cv-note").textContent = `${idx.length.toLocaleString()} region-weeks per map. Units: prices and payouts in $/MWh (weekly means), arbitrage in $/day; covariances are in the product of the two units. A dash means a measure is zero throughout the selection.`;
  }
  const pairSel = document.getElementById("cy-pair"), PAIRS = [];
  for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) PAIRS.push([p, q]);
  PAIRS.sort((a, b) => (b[1] === 3) - (a[1] === 3));
  pairSel.innerHTML = PAIRS.map(([p, q], j) => `<option value="${j}">${VARS[p][1]} × ${VARS[q][1]}</option>`).join("");
  pairSel.value = String(PAIRS.findIndex(([p, q]) => p === 2 && q === 3));
  function drawYears() {
    const [p, q] = PAIRS[+pairSel.value], rg = regionSel.value;
    const yrs = [...new Set(SC.raw.fy)].sort();
    const lab = y => `${y - 1}–${String(y).slice(2)}` + (y >= 2026 ? (per.value === "dev" ? " (Jul–Aug 2025 only)" : " (holdout)") : "");
    const rows = yrs.map(y => {
      const idx = SC.raw.fy.map((f, i) => (f === y && (rg === "all" || SC.raw.region[i] === +rg) && (per.value === "all" || isDev(i))) ? i : -1).filter(i => i >= 0);
      if (idx.length < 3) return "";
      const cells = MAPS.map(([k]) => { const v = stats(SC[k], idx).corr[p][q]; return `<td class="num">${isFinite(v) ? v.toFixed(2) : "–"}</td>`; }).join("");
      return `<tr><td>${lab(y)}</td><td class="num">${idx.length}</td>${cells}</tr>`;
    }).join("");
    document.getElementById("cy-table").innerHTML = `<thead><tr><th>Year</th><th class="num">Region-weeks</th>${MAPS.map(([, l]) => `<th class="num">${l}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  }
  for (const el of [yearSel, regionSel, mode, per]) el.addEventListener("change", draw);
  for (const el of [regionSel, per, pairSel]) el.addEventListener("change", drawYears);
  draw(); drawYears();
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
  --pos: #2a6fb0; --neg: #c2410c;
  --k1: #2a6fb0; --k2: #c2410c; --k3: #0f8a6e; --k4: #9d3ea8; --k5: #a37a00; --k6: #5b6764;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0;
  --pos: #4f8fd0; --neg: #e07a40;
  --k1: #6fa8e0; --k2: #f08a50; --k3: #47c4a2; --k4: #d07fdb; --k5: #e0bb45; --k6: #a3b0ac; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0;
  --pos: #4f8fd0; --neg: #e07a40;
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
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 14px; }}
.panel {{ position: relative; min-width: 0; }}
.panel h3 {{ font: 600 .95rem var(--display); margin: 0 0 4px; }}
.panel canvas {{ width: 100%; aspect-ratio: 1 / 0.85; max-width: 100%; display: block; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 4px; }}
.tip {{ position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: 6px 8px; font: 12px/1.4 var(--mono); box-shadow: 0 2px 8px rgb(0 0 0 / .15); white-space: nowrap; z-index: 2; }}
.combined {{ margin-bottom: .4rem; }}
.cm-year {{ margin: 1.1rem 0 .3rem; font: 600 .95rem var(--display); }}
.cm-year small {{ font: 400 .8rem var(--body); color: var(--muted); margin-left: 8px; }}
.cm-year .flag {{ color: var(--cap); }}
.cm-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 12px; }}
.cm-row figure {{ margin: 0; min-width: 0; overflow-x: auto; }}
.cm-row figcaption {{ font-size: .8rem; color: var(--muted); margin-bottom: 3px; }}
table.cm {{ width: auto; font-size: .78rem; }}
table.cm th, table.cm td {{ padding: 3px 6px; border-bottom: 0; text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
table.cm th {{ font-size: .68rem; letter-spacing: 0; text-transform: none; }}
table.cm td.self {{ color: var(--muted); }}
.combined canvas {{ aspect-ratio: 1 / 0.62; }}
.links {{ display: inline-flex; gap: 6px; align-items: center; font-size: .85rem; color: var(--muted); margin: 2px 0 6px; }}
.cv-block {{ margin: .9rem 0 1.2rem; }}
.cv-block h4 {{ font: 600 .9rem var(--display); margin: 0 0 4px; }}
.cv-block th code {{ text-transform: none; letter-spacing: 0; }}
.cv-block td.diag {{ font-weight: 500; color: var(--accent); }}
.marker {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }}
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

<h2>Week by week: arbitrage against what the week's prices pay</h2>
<p>Each point is one region-week (ISO weeks, Monday to Sunday, grouping trading days by their date): the battery's
perfect-foresight arbitrage value (x, the mean over the week's trading days, each solved separately) against a
measure of that week's prices (y, the mean of its daily values),
on the raw prices, on the mapped ones, and, for comparison, with every positive price scaled by the ratio of today's
cap to the cap in force then (negative prices aren't scaled, since the floor hasn't moved). A good map moves older days toward the recent pattern
without bending the relationship. Below $1,000 the chosen map changes nothing, so the payout above $1,000 is the
measure that shows the stretch most directly; the mean price and the $300 payout mix stretched and unchanged
prices. Both axes are symmetric-log. 2025–26 and 2026–27 are holdout years, shown for context.</p>
<div class="controls">
  <label for="sc-y">Y axis <select id="sc-y">
    <option value="cap1000">Payout above $1,000/MWh ($/MWh, weekly mean)</option>
    <option value="cap300">Payout above $300/MWh ($/MWh, weekly mean)</option>
    <option value="swap">Mean price ($/MWh)</option></select></label>
  <label for="sc-year">Year <select id="sc-year"></select></label>
  <label for="sc-region">Region <select id="sc-region"></select></label>
</div>
<div class="panel combined">
  <h3>All three on one figure</h3>
  <ul class="legend" id="sc-combined-legend"></ul>
  <label class="links" for="sc-links"><input type="checkbox" id="sc-links" checked> Join each region-week's three points</label>
  <canvas id="sc-combined" aria-label="Raw, $1,000-knee and constant-scaling points on one scatter"></canvas>
</div>
<p style="margin-top:1.2rem">The same points, one map per panel, coloured by year or region:</p>
<div id="sc-legend"></div>
<div class="panels">
  <div class="panel"><h3>Raw prices</h3><canvas id="sc-raw" aria-label="Scatter of daily arbitrage against the chosen price measure, raw prices"></canvas></div>
  <div class="panel"><h3>Mapped onto $23,200 (knee $1,000)</h3><canvas id="sc-mapped" aria-label="The same scatter on mapped prices"></canvas></div>
  <div class="panel"><h3>Constant scaling (for comparison)</h3><canvas id="sc-scaled" aria-label="The same scatter with prices scaled by the cap ratio"></canvas></div>
</div>
<p class="note" id="sc-note" style="color:var(--muted);font-size:.85rem"></p>

<h3 style="font:600 1.05rem var(--display);margin:1.8rem 0 .3rem">Covariance of the four weekly measures</h3>
<p>Sample covariance across region-weeks of the mean price, the payout above $300 (cap payout), the payout above $1,000
and the arbitrage value, on each set of prices. The Year and Region selectors above filter these too; by default only dev-period weeks are used, since 2025–26 and
2026–27 are holdout. Diagonal entries
are variances. Switch to correlation to compare across maps without the scale effect.</p>
<div class="controls"><label for="cv-mode">Show <select id="cv-mode">
  <option value="cov">Covariance</option><option value="corr">Correlation</option></select></label>
  <label for="cv-period">Period <select id="cv-period">
  <option value="dev">Dev weeks only (to Aug 2025)</option><option value="all">All weeks, incl. holdout</option></select></label></div>
<div id="cv-tables"></div>
<p class="note" id="cv-note" style="color:var(--muted);font-size:.85rem"></p>
<h4 style="font:600 .9rem var(--display);margin:1.2rem 0 4px">Correlation year by year</h4>
<p>One pair at a time, each financial year computed separately (Region and Period selectors apply; Year is ignored here).</p>
<div class="controls"><label for="cy-pair">Pair <select id="cy-pair"></select></label></div>
<div class="table-wrap"><table id="cy-table"></table></div>

<h2>Do the relationships hold year to year?</h2>
<p>Correlations between the four weekly measures (one value per region-week, as in the scatter above), computed
separately on each price set, so each map's own arbitrage values are correlated with its own price measures. One
row per financial year, plus all years together. They follow the Region selector above. A map that preserves the
market's structure should leave each year's pattern much as the raw prices have it, while bringing older years'
levels toward recent ones.</p>
<div class="controls">
  <label for="cm-method">Correlation <select id="cm-method">
    <option value="pearson">Pearson (values)</option><option value="spearman">Spearman (ranks)</option></select></label>
  <label for="cm-holdout"><input type="checkbox" id="cm-holdout"> Include holdout years (2025–26 from September, 2026–27)</label>
</div>
<p class="note" id="cm-note" style="color:var(--muted);font-size:.85rem"></p>
<div id="cm"></div>

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
    html += COVARIANCE_JS
    (OUT / "price_cap_report.html").write_text(html)
    print(OUT / "price_cap_report.html")


if __name__ == "__main__":
    main()
