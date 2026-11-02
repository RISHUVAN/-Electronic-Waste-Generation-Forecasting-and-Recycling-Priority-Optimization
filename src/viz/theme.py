"""Dark-only design system: CSS for Streamlit and a matching Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio

BG, SURFACE, SURFACE2, BORDER = "#0b1220", "#111b2e", "#16233a", "#223454"
TEXT, MUTED = "#e6edf6", "#93a5c1"
TEAL, EMERALD, LIME, AMBER, SKY, VIOLET, ROSE = "#2dd4bf", "#10b981", "#a3e635", "#fbbf24", "#38bdf8", "#a78bfa", "#fb7185"
COLORWAY = [TEAL, LIME, SKY, AMBER, VIOLET, EMERALD, ROSE]
SCALE = [[0, "#10323a"], [0.35, "#0f766e"], [0.65, "#2dd4bf"], [0.85, "#a3e635"], [1, "#fbbf24"]]
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif"
GRID = "rgba(147,165,193,0.14)"


def register_plotly():
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID, title=dict(font=dict(size=12, color=MUTED)))
    pio.templates["ewaste"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=TEXT, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY, xaxis=axis, yaxis=axis,
        hoverlabel=dict(bgcolor="#0b1220", bordercolor=TEAL, font=dict(color=TEXT, family=FONT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        coloraxis=dict(colorscale=SCALE, colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(color=MUTED)))))


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(45,212,191,.10), transparent 60%),
          radial-gradient(900px 500px at -10% 10%, rgba(16,185,129,.08), transparent 55%), {BG}; color: {TEXT}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
p, li, label, .stMarkdown {{ color: {TEXT}; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
a {{ color: {TEAL}; }}
*:focus-visible {{ outline: 2px solid {TEAL} !important; outline-offset: 2px; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0e1729 0%, #0a1020 100%); border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.brand {{ display:flex; gap:12px; align-items:center; padding: 4px 4px 16px 4px; border-bottom:1px solid {BORDER}; margin-bottom:14px; }}
.brand-mark {{ width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, {EMERALD}, {TEAL}); color:#04120f; font-weight:700; font-size:18px; flex:none; }}
.brand-name {{ font-weight:700; font-size:.82rem; letter-spacing:.08em; line-height:1.25; color:{TEXT}; }}
.brand-sub {{ font-size:.7rem; color:{MUTED}; margin-top:2px; }}
.side-label {{ font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; margin:18px 0 6px 2px; font-weight:600; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 7px 10px; border-radius: 9px; margin-bottom: 2px; width:100%; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(45,212,191,.08); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background: rgba(45,212,191,.14); box-shadow: inset 3px 0 0 {TEAL}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}

/* page header */
.page-head {{ margin: 0 0 18px 0; }}
.eyebrow {{ font-size:.7rem; letter-spacing:.16em; color:{TEAL}; font-weight:600; text-transform:uppercase; }}
.page-head h1 {{ font-size: 1.75rem; font-weight: 700; margin: 4px 0 4px 0; padding: 0; }}
.page-head p {{ color:{MUTED}; margin:0; font-size:.95rem; max-width: 880px; }}

/* KPI cards */
.kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin: 6px 0 18px 0; }}
.kpi {{ background: linear-gradient(160deg, {SURFACE2} 0%, {SURFACE} 100%); border:1px solid {BORDER}; border-radius:14px;
  padding:14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,.28); position:relative; overflow:hidden; }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: linear-gradient({TEAL}, {LIME}); }}
.kpi-label {{ font-size:.72rem; color:{MUTED}; letter-spacing:.06em; text-transform:uppercase; font-weight:600; }}
.kpi-value {{ font-size:1.55rem; font-weight:700; margin-top:4px; color:{TEXT}; line-height:1.2; }}
.kpi-sub {{ font-size:.76rem; color:{MUTED}; margin-top:3px; }}

/* callouts */
.callout {{ border:1px solid {BORDER}; border-left:4px solid {TEAL}; background: rgba(17,27,46,.85); border-radius:10px;
  padding:11px 14px; margin: 6px 0 14px 0; font-size:.88rem; color:{TEXT}; line-height:1.5; }}
.callout.warn {{ border-left-color:{AMBER}; }} .callout.danger {{ border-left-color:{ROSE}; }} .callout.ok {{ border-left-color:{LIME}; }}
.callout b {{ color:#fff; }}
.empty {{ text-align:center; padding:48px 20px; border:1px dashed {BORDER}; border-radius:14px; color:{MUTED}; background:rgba(17,27,46,.5); }}
.empty b {{ color:{TEXT}; display:block; margin-bottom:4px; font-size:1rem; }}

/* charts, tables, tabs, buttons */
[data-testid="stPlotlyChart"] {{ background: linear-gradient(180deg, {SURFACE} 0%, #0e1729 100%); border:1px solid {BORDER};
  border-radius:14px; padding:6px 8px; box-shadow: 0 6px 18px rgba(0,0,0,.25); }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
button[data-baseweb="tab"] {{ color:{MUTED}; font-weight:600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{TEAL}; }}
div[data-baseweb="tab-highlight"] {{ background-color:{TEAL}; }}
.stDownloadButton button, .stButton button {{ background: linear-gradient(135deg, {EMERALD}, #0d9488); color:#04120f; border:0;
  font-weight:600; border-radius:10px; padding:.5rem 1rem; }}
.stDownloadButton button:hover {{ filter: brightness(1.1); color:#04120f; }}
details {{ border:1px solid {BORDER} !important; border-radius:12px !important; background: rgba(17,27,46,.6); }}

@media (max-width: 768px) {{
  .block-container {{ padding: 1rem .8rem 2rem .8rem; }}
  .page-head h1 {{ font-size: 1.4rem; }} .kpi-value {{ font-size: 1.25rem; }}
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); gap:10px; }}
}}
</style>
"""
