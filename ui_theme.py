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
          @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Space+Mono:wght@400;700&display=swap');

          :root{
            --bg:#0E1117; --panel:#111520; --muted:#1A2030; --text:#E8EAED; --text-dim:#B5BDC9;
            --brand:#6C9FF7; --brand-2:#9A7DFF; --ok:#50C878; --warn:#FFC857; --bad:#FF6B6B;
            --radius:16px; --shadow:0 10px 30px rgba(0,0,0,.35); --shadow-soft:0 6px 18px rgba(0,0,0,.28);
          }

          [data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 600px at 20% -10%, rgba(108,159,247,0.07), transparent 60%),
                        radial-gradient(1200px 600px at 120% 20%, rgba(154,125,255,0.06), transparent 60%),
                        var(--bg);
          }

          html, body, [data-testid="stAppViewContainer"] * {
            -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
          }
          body, p, li, label, span, div { color: var(--text); font-size: 16.5px; line-height: 1.6; }

          h1, h2, h3, .app-hero__title {
            font-family: "Playfair Display", serif !important; letter-spacing: .2px;
          }
          h1 { font-size: 42px; font-weight: 700; margin: 0 0 14px 0; }
          h2 { font-size: 30px; font-weight: 700; margin: 0 0 12px 0; }
          h3 { font-size: 22px; font-weight: 700; margin: 0 0 10px 0; }

          code, kbd, samp, .mono, [data-testid="stMetricValue"] {
            font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
          }

          .main .block-container { padding: 24px 42px 56px 42px; }

          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17,21,32,0.95) 0%, rgba(17,21,32,0.85) 100%), var(--panel);
            border-right: 1px solid rgba(255,255,255,0.06);
          }

          .card {
            background: var(--panel); border-radius: var(--radius); box-shadow: var(--shadow-soft);
            border: 1px solid rgba(255,255,255,0.06); padding: 22px 22px;
          }
          .card + .card { margin-top: 18px; }

          .app-hero {
            border-radius: calc(var(--radius) + 2px); padding: 28px 28px;
            background: linear-gradient(120deg, rgba(108,159,247,0.20), rgba(154,125,255,0.18)) border-box, var(--panel);
            border: 1px solid rgba(255,255,255,0.10); box-shadow: var(--shadow); margin-bottom: 22px;
          }
          .app-hero__title {
            font-size: 44px; margin: 0 0 8px 0;
            background: linear-gradient(90deg, var(--brand), var(--brand-2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          }
          .app-hero__subtitle { color: var(--text-dim); font-size: 18px; margin: 0; }

          [data-baseweb="tab-list"] { gap: 6px; }
          [data-baseweb="tab"] {
            background: var(--panel); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 1000px; padding: 10px 16px; color: var(--text-dim); transition: all .2s ease;
          }
          [data-baseweb="tab"][aria-selected="true"] {
            color: var(--text); border-color: rgba(255,255,255,0.16); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
          }

          [data-testid="stMetric"] {
            background: var(--panel); border: 1px solid rgba(255,255,255,0.06);
            border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow-soft);
          }
          [data-testid="stMetricLabel"] { color: var(--text-dim); font-size: 13px; }
          [data-testid="stMetricValue"] { font-size: 28px; }

          .stButton > button {
            background: linear-gradient(90deg, var(--brand), var(--brand-2)); color: #0B0E14; border: 0; border-radius: 12px;
            padding: 10px 16px; font-weight: 700; letter-spacing: .2px; box-shadow: 0 6px 16px rgba(108,159,247,0.25);
          }
          .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(108,159,247,0.35); }

          .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background: var(--muted); border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); color: var(--text);
          }

          details[data-testid="stExpander"] {
            background: var(--panel); border-radius: var(--radius); border: 1px solid rgba(255,255,255,0.08); padding: 6px 10px;
          }

          .stDataFrame, .stTable {
            border: 1px solid rgba(255,255,255,0.06); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow-soft);
          }

          .recommendation-card {
            background: var(--panel); border-radius: var(--radius);
            border: 1px solid rgba(255,255,255,0.08); padding: 22px; text-align: center; box-shadow: var(--shadow-soft);
          }
          .recommendation-card h2 {
            font-family: "Playfair Display", serif !important; font-size: 28px; margin: 0 0 8px 0;
          }
          .recommendation-card p {
            color: var(--text-dim); margin: 0; font-family: "Space Mono", ui-monospace, monospace !important;
          }
          .high-conviction { border-left: 6px solid var(--ok); }
          .medium-conviction { border-left: 6px solid var(--warn); }
          .low-conviction { border-left: 6px solid var(--bad); }

          .section { padding: 10px 0 4px 0; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 18px; }
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
