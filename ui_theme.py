from __future__ import annotations
import streamlit as st
from contextlib import contextmanager

def apply_theme(page_title: str = "Startup Evaluator", page_icon: str | None = "✨"):
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          :root{
            --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
                         "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --font-mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Monaco, Consolas, monospace;

            --bg: #F6F8FB;
            --ink: #0F172A;
            --ink-dim: #475569;
            --line: rgba(15, 23, 42, 0.08);

            --glass: rgba(255, 255, 255, 0.55);
            --glass-strong: rgba(255, 255, 255, 0.70);
            --glass-field: rgba(255, 255, 255, 0.86);
            --glass-menu: rgba(255, 255, 255, 0.98);
            --glass-border: rgba(255, 255, 255, 0.95);

            --accent: #0A84FF;
            --ok: #34C759; --warn: #FFCC00; --bad: #FF3B30;

            --radius: 16px;
            --radius-lg: 18px;
            --radius-sm: 12px;
            --shadow: 0 18px 40px rgba(15,23,42,0.08);
            --shadow-soft: 0 10px 26px rgba(15,23,42,0.07);
            --blur: saturate(180%) blur(16px);
            --focus: 0 0 0 3px rgba(10,132,255,0.25);
          }

          [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(900px 500px at 10% -10%, rgba(10,132,255,0.10), transparent 60%),
              radial-gradient(800px 400px at 120% 0%, rgba(100,210,255,0.10), transparent 65%),
              var(--bg);
          }

          html, body, [data-testid="stAppViewContainer"] * {
            -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
          }
          body, p, li, label, span, div {
            color: var(--ink);
            font-family: var(--font-sans);
            font-size: 16px;
            line-height: 1.55;
          }
          h1, h2, h3 {
            font-weight: 700;
            letter-spacing: .1px;
            margin: 0 0 10px 0;
            color: var(--ink);
          }
          h1 { font-size: 40px; }
          h2 { font-size: 28px; }
          h3 { font-size: 20px; }
          code, .mono, [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }

          .main .block-container { padding: 28px 42px 60px 42px; }

          .card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur);
            backdrop-filter: var(--blur);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            padding: 22px 22px;
            overflow: hidden;
          }
          .card + .card { margin-top: 18px; }

          .app-hero {
            border-radius: 22px;
            background: var(--glass-strong);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur);
            backdrop-filter: var(--blur);
            box-shadow: var(--shadow);
            padding: 28px 28px;
            margin: 4px 0 22px 0;
          }
          .app-hero__title {
            font-size: 44px; font-weight: 750; margin: 0 0 6px 0;
            background: linear-gradient(90deg, var(--ink) 0%, #1F2937 60%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          }
          .app-hero__subtitle { color: var(--ink-dim); font-size: 18px; margin: 0; }

          [data-baseweb="tab-list"] { gap: 8px; }
          [data-baseweb="tab"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            color: var(--ink-dim);
            border-radius: 999px; padding: 10px 16px;
            box-shadow: var(--shadow-soft);
            transition: transform .15s ease, box-shadow .15s ease, color .15s ease;
          }
          [data-baseweb="tab"][aria-selected="true"] {
            color: var(--ink);
            border-color: rgba(255,255,255,1);
            box-shadow: 0 8px 20px rgba(10,132,255,0.12);
          }

          .stButton > button {
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            color: var(--ink);
            border-radius: 14px; padding: 10px 16px; font-weight: 700;
            box-shadow: var(--shadow-soft);
            transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
          }
          .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(10,132,255,0.22);
            border-color: rgba(10,132,255,0.65);
          }
          .stButton > button:focus { outline: none !important; box-shadow: var(--focus) !important; }

          .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background: var(--glass-field) !important;
            border: 1px solid var(--glass-border) !important;
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: 12px !important;
            color: var(--ink) !important;
            caret-color: var(--accent) !important;
          }
          .stTextInput input::placeholder, .stNumberInput input::placeholder, .stTextArea textarea::placeholder { color: #94A3B8 !important; }
          .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            outline: none !important; border-color: rgba(10,132,255,0.75) !important; box-shadow: var(--focus) !important;
          }

          /* Selectbox fix: visible text, frosted backgrounds, no corner bleed */
          .stSelectbox > div { background: transparent !important; border: none !important; padding: 0 !important; border-radius: 12px !important; overflow: hidden !important; }
          .stSelectbox div[data-baseweb="select"] > div:first-child,
          .stSelectbox div[role="combobox"] {
            background: var(--glass-field) !important;
            border: 1px solid var(--glass-border) !important;
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            color: var(--ink) !important;
            border-radius: 12px !important;
          }
          .stSelectbox div[data-baseweb="select"]:focus-within > div:first-child,
          .stSelectbox div[role="combobox"]:focus-within {
            outline: none !important; border-color: rgba(10,132,255,0.75) !important; box-shadow: var(--focus) !important;
          }
          .stSelectbox * { color: var(--ink) !important; }
          .stSelectbox svg, .stSelectbox svg path { stroke: var(--ink) !important; fill: var(--ink) !important; }
          .stSelectbox [data-baseweb="popover"] [data-baseweb="menu"] {
            background: var(--glass-menu) !important;
            border: 1px solid var(--glass-border) !important;
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: 14px !important;
            box-shadow: var(--shadow);
            color: var(--ink) !important;
            overflow: hidden !important;
          }
          .stSelectbox [data-baseweb="menu"] [role="option"] { color: var(--ink) !important; }
          .stSelectbox [data-baseweb="menu"] [role="option"]:hover { background: rgba(10,132,255,0.08) !important; }
          .stSelectbox [data-baseweb="menu"] [role="option"][aria-selected="true"] { background: rgba(10,132,255,0.14) !important; color: var(--ink) !important; }

          .stSlider > div [role="slider"] { background: var(--accent) !important; box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important; }
          .stSlider > div [data-baseweb="slider"]>div { background: linear-gradient(90deg, rgba(10,132,255,0.25), rgba(100,210,255,0.25)) !important; }

          [data-testid="stMetric"] {
            background: rgba(255,255,255,0.6);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: var(--shadow-soft);
          }
          [data-testid="stMetricLabel"] { color: var(--ink-dim); font-size: 13px; }
          [data-testid="stMetricValue"] { font-size: 28px; color: var(--ink); }

          details[data-testid="stExpander"],
          .stDataFrame, .stTable,
          [data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.6);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow-soft);
          }

          .recommendation-card {
            background: rgba(255,255,255,0.6);
            border: 1px solid var(--glass-border);
            -webkit-backdrop-filter: var(--blur); backdrop-filter: var(--blur);
            border-radius: var(--radius);
            padding: 22px; text-align: center; box-shadow: var(--shadow-soft);
          }
          .recommendation-card h2 { font-size: 28px; margin: 0 0 8px 0; }
          .recommendation-card p { color: var(--ink-dim); margin: 0; font-family: var(--font-mono) !important; }

          .high-conviction { border-left: 6px solid var(--ok); }
          .medium-conviction { border-left: 6px solid var(--warn); }
          .low-conviction { border-left: 6px solid var(--bad); }

          .section { padding: 10px 0 4px 0; border-top: 1px solid var(--line); margin-top: 18px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="app-hero">
          <div class="app-hero__title">{title}</div>
          <p class="app-hero__subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_heading(text: str, sub: str | None = None):
    st.markdown(
        f"""
        <div class="section">
          <h2>{text}</h2>
          {f'<p class="app-hero__subtitle" style="margin-top:-4px">{sub}</p>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

@contextmanager
def card():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown('</div>', unsafe_allow_html=True)
