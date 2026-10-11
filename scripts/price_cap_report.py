"""Render scripts/price_cap_analysis.py's results as one HTML page.

    uv run python scripts/price_cap_report.py   # -> data/analysis/price_cap_report.html
"""

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"

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
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #121816; --panel: #1a2220; --ink: #e5ebe8; --muted: #9aa7a3; --rule: #2e3a37;
  --accent: #4fc1b0; --cap: #e88a52; --s1: #7d8a86; --s2: #e88a52; --s3: #e0b85a; --s4: #4fc1b0; color-scheme: dark }}
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
    (OUT / "price_cap_report.html").write_text(html)
    print(OUT / "price_cap_report.html")


if __name__ == "__main__":
    main()
