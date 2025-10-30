import contextlib
import streamlit as st

def apply_theme(page_title: str = "App", page_icon: str = "🔎"):
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide", initial_sidebar_state="collapsed")
    # Font and base styles (smaller, consistent)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: Inter, -apple-system, Segoe UI, Roboto, system-ui, sans-serif !important;
            font-size: 14px !important;
            color: #0f1221 !important;
        }
        h1, h2, h3 { font-weight: 600; }
        /* Card shell */
        .ah-card {
            background: #fff;
            border: 1px solid rgba(15,18,33,0.06);
            border-radius: 12px;
            padding: 10px 12px;
            box-shadow: 0 1px 2px rgba(15,18,33,0.04);
        }
        .ah-hero {
            padding: 6px 8px 6px 0;
        }
        .ah-section h2 {
            margin: 0 0 4px 0 !important;
            padding: 0;
        }
        .ah-section small {
            color: #667085;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero(title: str, subtitle: str = ""):
    st.markdown(f'<div class="ah-hero"><h1>{title}</h1><small>{subtitle}</small></div>', unsafe_allow_html=True)

def section_heading(title: str, subtitle: str = None):
    if subtitle:
        st.markdown(f'<div class="ah-section"><h2>{title}</h2><small>{subtitle}</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ah-section"><h2>{title}</h2></div>', unsafe_allow_html=True)

@contextlib.contextmanager
def card():
    c = st.container()
    with c:
        st.markdown('<div class="ah-card">', unsafe_allow_html=True)
        yield
        st.markdown('</div>', unsafe_allow_html=True)
