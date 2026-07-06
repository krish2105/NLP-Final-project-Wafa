"""Presentation-layer helpers for the Streamlit dashboard.

Pure UI: a light/dark theme engine (CSS-variable palettes injected into the
Streamlit chrome), plus small HTML component helpers (badges, stat cards, themed
tables, framed figures). No business logic lives here — it only renders things the
`src/` logic modules produce, so `app.py` stays a thin presentation layer.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #
LIGHT = {
    "bg": "#eef2f7",
    "bg2": "#e6ecf3",
    "surface": "#ffffff",
    "surface2": "#f5f8fc",
    "text": "#0f1c2e",
    "muted": "#5b6b7f",
    "border": "#e2e8f0",
    "primary": "#0f4c81",
    "primary2": "#1a6fb0",
    "shadow": "0 1px 3px rgba(15,30,50,.10), 0 8px 24px rgba(15,30,50,.06)",
    "input_bg": "#ffffff",
}
DARK = {
    "bg": "#0b1220",
    "bg2": "#0e1626",
    "surface": "#131d2e",
    "surface2": "#1a2637",
    "text": "#e7eef8",
    "muted": "#9fb2c6",
    "border": "#26344a",
    "primary": "#3b82f6",
    "primary2": "#60a5fa",
    "shadow": "0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35)",
    "input_bg": "#0f1a2b",
}
# Shared accent colours (readable on both themes)
ACCENT = {"high": "#e5484d", "medium": "#e0a02c", "low": "#2fa36b", "leaver": "#7c5cff"}


def palette(dark: bool) -> dict:
    return DARK if dark else LIGHT


# --------------------------------------------------------------------------- #
# Theme CSS
# --------------------------------------------------------------------------- #
def theme_css(dark: bool) -> str:
    p = palette(dark)
    grad = (
        "linear-gradient(120deg,#0b3a63 0%,#0f4c81 45%,#1a6fb0 100%)"
        if not dark
        else "linear-gradient(120deg,#0a1626 0%,#12233b 50%,#1b3a5c 100%)"
    )
    header_border = "rgba(255,255,255,.12)"
    return f"""
<style>
:root {{
  --bg:{p['bg']}; --bg2:{p['bg2']}; --surface:{p['surface']}; --surface2:{p['surface2']};
  --text:{p['text']}; --muted:{p['muted']}; --border:{p['border']};
  --primary:{p['primary']}; --primary2:{p['primary2']}; --shadow:{p['shadow']};
  --input-bg:{p['input_bg']};
  --high:{ACCENT['high']}; --medium:{ACCENT['medium']}; --low:{ACCENT['low']}; --leaver:{ACCENT['leaver']};
}}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
  font-family: 'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
}}

/* Page + main surface */
.stApp, [data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 600px at 100% -10%, var(--bg2), transparent 60%),
    var(--bg);
  color: var(--text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Typography */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6 {{ color: var(--text); }}
.stApp p, .stApp li, .stApp label, .stApp span, .stMarkdown {{ color: var(--text); }}
[data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--muted) !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] * {{ color: var(--text); }}

/* Header banner */
.wafa-header {{
  background: {grad};
  padding: 1.3rem 1.6rem; border-radius: 16px; color:#fff; margin-bottom: 1.2rem;
  border: 1px solid {header_border};
  box-shadow: 0 10px 30px rgba(10,40,70,.28);
}}
.wafa-header h1 {{ margin:0; font-size:1.55rem; color:#fff; letter-spacing:.2px; }}
.wafa-header p {{ margin:.25rem 0 0 0; opacity:.9; font-size:.9rem; color:#eaf2fb; }}
.wafa-header .chips {{ margin-top:.7rem; display:flex; gap:.5rem; flex-wrap:wrap; }}
.wafa-chip {{
  background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  color:#fff; padding:.18rem .6rem; border-radius:999px; font-size:.74rem; font-weight:600;
}}

/* Cards */
.wafa-card {{
  background: var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:1.15rem 1.3rem; margin-bottom:1rem; box-shadow: var(--shadow);
  border-top: 3px solid var(--primary2);
  height: 100%;
}}
.wafa-card h4 {{ margin:0 0 .6rem 0; font-size:1.02rem; display:flex; align-items:center; gap:.4rem; }}
.wafa-card .row {{ display:flex; justify-content:space-between; gap:.5rem; padding:.18rem 0;
                  border-bottom:1px dashed var(--border); font-size:.9rem; }}
.wafa-card .row:last-child {{ border-bottom:none; }}
.wafa-card .row .k {{ color: var(--muted); }}
.wafa-card .row .v {{ color: var(--text); font-weight:600; text-align:right; }}
.wafa-card ul {{ margin:.3rem 0 0 0; padding-left:1.1rem; }}
.wafa-card ul li {{ margin:.12rem 0; font-size:.9rem; }}

/* Badges + pills */
.badge {{ display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700;
         font-size:.78rem; color:#fff; letter-spacing:.2px; white-space:nowrap; }}
.badge-high{{background:var(--high);}} .badge-medium{{background:var(--medium);}}
.badge-low{{background:var(--low);}} .badge-leaver{{background:var(--leaver);}}
.badge-soft {{ background: color-mix(in srgb, var(--primary) 16%, transparent);
              color: var(--primary2); border:1px solid var(--border); }}

/* Stat cards */
.stat-grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
             gap:.8rem; margin:.2rem 0 1rem 0; }}
.stat {{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
        padding:.9rem 1.1rem; box-shadow: var(--shadow); }}
.stat .label {{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.6px; }}
.stat .value {{ font-size:1.7rem; font-weight:800; color:var(--text); line-height:1.2; }}
.stat .sub {{ color:var(--muted); font-size:.78rem; }}
.stat.accent-high {{ border-left:4px solid var(--high); }}
.stat.accent-med {{ border-left:4px solid var(--medium); }}
.stat.accent-primary {{ border-left:4px solid var(--primary2); }}

/* Section heading */
.wafa-section {{ font-size:1.05rem; font-weight:700; color:var(--text);
                margin:1.1rem 0 .5rem 0; display:flex; align-items:center; gap:.5rem; }}
.wafa-section:before {{ content:""; width:4px; height:1.05rem; border-radius:3px; background:var(--primary2); }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid var(--border);
                                     background:transparent; }}
.stTabs [data-baseweb="tab"] {{
  background: var(--surface2); border:1px solid var(--border); border-bottom:none;
  border-radius:10px 10px 0 0; padding:.5rem .9rem; color:var(--muted); font-weight:600;
}}
.stTabs [aria-selected="true"] {{ background: var(--surface); color: var(--primary2);
                                  border-top:2px solid var(--primary2); }}

/* Buttons */
.stButton > button {{
  border-radius:10px; border:1px solid var(--border); background:var(--surface2);
  color:var(--text); font-weight:600; transition:all .15s ease;
}}
.stButton > button:hover {{ border-color:var(--primary2); color:var(--primary2);
                            transform: translateY(-1px); }}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg,var(--primary),var(--primary2)); color:#fff;
  border:none; box-shadow:0 6px 16px rgba(20,80,140,.35);
}}
.stDownloadButton > button {{ border-radius:10px; border:1px solid var(--border);
                              background:var(--surface2); color:var(--text); }}

/* Inputs */
.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
  background: var(--input-bg) !important; color: var(--text) !important;
  border-radius:10px !important; border:1px solid var(--border) !important;
}}
.stTextArea textarea {{ font-size:.92rem; }}

/* Metrics (native, used as fallback) */
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border);
                            border-radius:14px; padding:.8rem 1rem; box-shadow:var(--shadow); }}
[data-testid="stMetricValue"] {{ color:var(--text); }}
[data-testid="stMetricLabel"] {{ color:var(--muted); }}

/* Alerts – themed but keep semantic tint */
[data-testid="stAlert"] {{ border-radius:12px; border:1px solid var(--border); }}

/* Real bordered containers (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--surface); border:1px solid var(--border) !important;
  border-radius:14px; box-shadow: var(--shadow); padding:.4rem .6rem;
}}
/* keep the sidebar's own block wrappers transparent */
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
  box-shadow:none; }}

/* Themed HTML tables */
.wafa-table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:14px;
                   box-shadow:var(--shadow); background:var(--surface); }}
table.wafa-table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
table.wafa-table thead th {{ position:sticky; top:0; background:var(--surface2);
   color:var(--muted); text-align:left; padding:.55rem .7rem; font-weight:700;
   border-bottom:1px solid var(--border); white-space:nowrap; text-transform:uppercase;
   font-size:.72rem; letter-spacing:.4px; }}
table.wafa-table tbody td {{ padding:.5rem .7rem; border-bottom:1px solid var(--border);
   color:var(--text); white-space:nowrap; }}
table.wafa-table tbody tr:nth-child(even) {{ background:var(--surface2); }}
table.wafa-table tbody tr:hover {{ background: color-mix(in srgb, var(--primary) 10%, transparent); }}
.pill {{ padding:.12rem .55rem; border-radius:999px; font-size:.74rem; font-weight:700; color:#fff; white-space:nowrap; }}
.pill-high{{background:var(--high);}} .pill-medium{{background:var(--medium);}}
.pill-low{{background:var(--low);}}

/* Figure frames (keep PNGs readable in dark mode) */
.fig-card {{ background:#ffffff; border:1px solid var(--border); border-radius:14px;
            padding:.6rem; box-shadow:var(--shadow); margin-bottom:.6rem; }}
.fig-card img {{ width:100%; border-radius:8px; display:block; }}
.fig-cap {{ text-align:center; color:var(--muted); font-size:.78rem; margin-top:.35rem; }}

/* Responsive tweaks */
@media (max-width: 640px) {{
  .wafa-header h1 {{ font-size:1.2rem; }}
  .block-container {{ padding-left:.6rem; padding-right:.6rem; }}
}}
</style>
"""


# --------------------------------------------------------------------------- #
# HTML component helpers
# --------------------------------------------------------------------------- #
def header_html(generator: str, theme_label: str) -> str:
    return f"""
<div class="wafa-header">
  <h1>🏦 Project Wafa — Retention Console</h1>
  <p>Falcon Bank UAE · Listen → Understand → Act · Every draft is human-reviewed, nothing auto-sends</p>
  <div class="chips">
    <span class="wafa-chip">🌍 EN · AR · HI · TL</span>
    <span class="wafa-chip">🧠 DistilBERT + TF-IDF</span>
    <span class="wafa-chip">✉️ {generator}</span>
    <span class="wafa-chip">🛡️ Guardrailed · Human-in-loop</span>
  </div>
</div>
"""


def badge(band: str, leaver: bool = False) -> str:
    if leaver:
        return '<span class="badge badge-leaver">Dignified Goodbye</span>'
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(band, "badge-low")
    return f'<span class="badge {cls}">{band}</span>'


def pill(band: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}.get(band, "pill-low")
    return f'<span class="pill {cls}">{band}</span>'


def stat_card(label: str, value, sub: str = "", accent: str = "primary") -> str:
    cls = {"high": "accent-high", "med": "accent-med", "primary": "accent-primary"}.get(accent, "accent-primary")
    return (
        f'<div class="stat {cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def stat_grid(cards: list[str]) -> str:
    return f'<div class="stat-grid">{"".join(cards)}</div>'


def section(title: str) -> str:
    return f'<div class="wafa-section">{title}</div>'


def render_table(df: pd.DataFrame, height: int = 360, band_col: Optional[str] = None) -> str:
    """Return a fully-themed, scrollable HTML table (works in light AND dark)."""
    d = df.copy()
    if band_col and band_col in d.columns:
        d[band_col] = d[band_col].map(lambda b: pill(str(b)))
    html = d.to_html(index=False, escape=False, border=0, classes="wafa-table")
    return f'<div class="wafa-table-wrap" style="max-height:{height}px">{html}</div>'


def figure_card(path: Path, caption: str = "") -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode()
    cap = f'<div class="fig-cap">{caption}</div>' if caption else ""
    return f'<div class="fig-card"><img src="data:image/png;base64,{b64}"/>{cap}</div>'


def plotly_layout(dark: bool, height: int = 360) -> dict:
    p = palette(dark)
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text"], family="Inter, Segoe UI, sans-serif"),
        height=height,
        margin=dict(t=30, r=10, l=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
