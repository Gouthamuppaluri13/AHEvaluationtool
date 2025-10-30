import contextlib
import streamlit as st

def apply_theme(page_title: str = "App", page_icon: str = "🔎"):
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root{
          --ah-bg: #F6F7FB;
          --ah-fg: #0F1221;
          --ah-muted: #667085;
          --ah-border: rgba(15,18,33,0.10);
          --ah-card: #FFFFFF;
          --ah-primary: #0A84FF;
          --ah-accent: #6E56CF;
          --ah-success: #34C759;
          --ah-warn: #FF9F0A;
          --ah-danger: #FF3B30;

          --ah-space-1: 4px;
          --ah-space-2: 8px;
          --ah-space-3: 12px;
          --ah-space-4: 16px;
          --ah-space-5: 20px;
          --ah-space-6: 24px;
          --ah-radius: 12px;

          --ah-h1: 28px;
          --ah-h2: 20px;
          --ah-h3: 16px;
          --ah-body: 14px;
          --ah-small: 12px;
        }

        html, body, [class*="css"] {
          font-family: Inter, -apple-system, Segoe UI, Roboto, system-ui, sans-serif !important;
          font-size: var(--ah-body) !important;
          color: var(--ah-fg) !important;
          background:
            radial-gradient(1200px 600px at -10% -20%, #C7D7FF1A, transparent 60%),
            radial-gradient(1000px 500px at 110% -10%, #FFD1DC1A, transparent 50%),
            var(--ah-bg) !important;
        }

        /* Streamlit containers + base spacing */
        .block-container { padding-top: 8px !important; padding-bottom: 16px !important; max-width: 1200px; }
        [data-testid="stVerticalBlock"] { gap: 10px !important; }
        [data-testid="stHorizontalBlock"] { gap: 12px !important; }

        /* Typography */
        .ah-hero h1 { font-size: var(--ah-h1); font-weight: 700; margin: 0; letter-spacing: -0.01em; }
        .ah-hero small { color: var(--ah-muted); font-size: var(--ah-small); }

        .ah-section h2 { font-size: var(--ah-h2); font-weight: 700; margin: 0; letter-spacing: -0.01em; }
        .ah-section small { color: var(--ah-muted); font-size: var(--ah-small); display:block; margin-top: 2px; }

        .ah-subtle { color: var(--ah-muted); }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
          margin: 0 0 8px 0 !important;
          color: var(--ah-fg) !important;
          letter-spacing: -0.01em;
        }
        .stMarkdown h1 { font-size: var(--ah-h1) !important; }
        .stMarkdown h2 { font-size: var(--ah-h2) !important; }
        .stMarkdown h3 { font-size: var(--ah-h3) !important; color: #2F334D !important; }
        .stMarkdown p, .stMarkdown li { font-size: 13.5px !important; line-height: 1.28rem !important; }

        /* Headings divider line to separate sections */
        .ah-section { padding-top: 6px; border-top: 1px solid rgba(15,18,33,0.06); }

        /* Card */
        .ah-card {
          background: var(--ah-card);
          border: 1px solid var(--ah-border);
          border-radius: var(--ah-radius);
          padding: 12px 14px;
          box-shadow: 0 6px 20px rgba(15,18,33,0.05), 0 1px 2px rgba(15,18,33,0.05);
        }
        .ah-card-header {
          display:flex; align-items:center; gap:8px; margin-bottom:6px;
        }
        .ah-card-header .ah-dot {
          width: 8px; height: 8px; border-radius: 999px; background: var(--ah-primary);
          box-shadow: 0 0 0 3px rgba(10,132,255,0.15);
        }
        .ah-card-header h3 { font-size: var(--ah-h3); font-weight: 700; margin: 0; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 6px !important; }
        .stTabs [data-baseweb="tab"] {
          padding: 6px 10px !important; font-size: 13px !important; border-radius: 8px !important;
          background: #fff; border: 1px solid rgba(15,18,33,0.08);
        }

        /* Buttons and inputs */
        .stButton > button { padding: 8px 12px !important; border-radius: 10px !important; font-size: 13px !important; }
        .stTextInput, .stSelectbox, .stTextArea, .stNumberInput, .stSlider { margin-bottom: 8px !important; }

        /* Metrics */
        div[data-testid="stMetric"] { margin-bottom: 6px !important; }

        /* Expanders */
        div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid rgba(15,18,33,0.06); }

        /* Tables/DataFrame */
        .stDataFrame, .stTable { font-size: 13px; }

        /* Header strip */
        header[data-testid="stHeader"] { height: 0px; padding: 0; background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero(title: str, subtitle: str = ""):
    st.markdown(
        f'''
        <div class="ah-hero">
          <h1>{title}</h1>
          <small>{subtitle}</small>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def section_heading(title: str, subtitle: str = None):
    if subtitle:
        st.markdown(f'<div class="ah-section"><h2>{title}</h2><small>{subtitle}</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ah-section"><h2>{title}</h2></div>', unsafe_allow_html=True)

@contextlib.contextmanager
def card(title: str | None = None, subtitle: str | None = None, accent: str = "primary"):
    c = st.container()
    with c:
        st.markdown('<div class="ah-card">', unsafe_allow_html=True)
        if title:
            st.markdown(
                f'''
                <div class="ah-card-header">
                    <span class="ah-dot" style="background: var(--ah-{accent});"></span>
                    <h3>{title}</h3>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            if subtitle:
                st.markdown(f'<div class="ah-subtle" style="margin-top:-2px;margin-bottom:6px;">{subtitle}</div>', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)
