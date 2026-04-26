from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

C_PRIMARY = "#FADB5F"
C_DEEP_GOLD = "#D4AF37"
C_NAVY_SURFACE = "#121E38"
C_SOFT_GOLD = "#FDF0B5"
C_GREEN = "#00E676"
C_RED = "#FF1744"
C_MUTED = "#E6E8E6"
C_BRIGHT_BLUE = "#4FC3F7"
C_VOL_UP = "#FADB5F"
C_VOL_DOWN = "#4FC3F7"

PAGES = [
    ("📂", "Data Loader"),
    ("📊", "Explore Data with AI Insights"),
    ("🔮", "Forecasting with AI Insights"),
]

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#FFFFFF", size=12),
    xaxis=dict(
        gridcolor="#162140",
        linecolor="#253658",
        title_font=dict(color="#FFFFFF"),
        tickfont=dict(color="#FFFFFF"),
    ),
    yaxis=dict(
        gridcolor="#162140",
        linecolor="#253658",
        title_font=dict(color="#FFFFFF"),
        tickfont=dict(color="#FFFFFF"),
    ),
    legend=dict(font=dict(color="#FFFFFF"), bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=55, b=40, l=10, r=10),
    colorway=[C_PRIMARY, C_GREEN, C_SOFT_GOLD, C_RED, C_BRIGHT_BLUE, "#00FFD1", C_DEEP_GOLD, "#FFFFFF"],
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Bitcoin Analytics Lab",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --accent      : #FADB5F;
    --accent-dim  : #D4AF37;
    --bg-base     : #080E1E;
    --bg-card     : #0F1A30;
    --bg-panel    : #162140;
    --bg-border   : #253658;
    --text-main   : #FFFFFF;
    --text-muted  : #A8B8D0;
    --font-mono   : 'JetBrains Mono', monospace;
    --font-main   : 'Space Grotesk', sans-serif;
}

*, *::before, *::after {
    box-sizing: border-box;
}

html {
    scrollbar-gutter: stable;
}

body {
    overflow-y: scroll;
}

[data-testid="stAppViewContainer"] {
    scrollbar-gutter: stable both-edges;
    overflow-y: scroll !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1.6rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

html, body, .stApp, .main,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg-base) !important;
    font-family: var(--font-main) !important;
    color: var(--text-main);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -20%;
    right: -15%;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(250,219,95,0.07) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--bg-border) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}

.page-header {
    background: linear-gradient(135deg, var(--bg-panel) 0%, rgba(22,33,64,0.4) 100%);
    border: 1px solid var(--bg-border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-weight: 700;
    font-size: 1.75rem;
    margin: 0 0 0.35rem;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}
.page-header h1 span { color: var(--accent); }
.page-header p { color: var(--text-muted); margin: 0; font-size: 0.9rem; }

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-label {
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}
.kpi-value {
    color: var(--accent);
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0.2rem 0 0;
    font-family: var(--font-mono);
    overflow-wrap: break-word;
}

code {
    background: rgba(250, 219, 95, 0.08) !important;
    color: var(--accent) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.3rem !important;
    font-family: var(--font-mono) !important;
}
.kpi-sub {
    color: var(--text-muted);
    font-size: 0.73rem;
    margin-top: 0.25rem;
}

.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--bg-border);
}

.stButton > button {
    width: 100%;
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-family: var(--font-main) !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(250,219,95,0.06) !important;
}
button[kind="primary"] {
    background: var(--accent) !important;
    color: #000000 !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background: var(--accent-dim) !important;
    color: #000000 !important;
}

label {
    color: var(--text-main) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: var(--font-main) !important;
}

/* Data source selector: improve readability and active-state visibility */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 0.6rem;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 9px !important;
    padding: 0.45rem 0.75rem !important;
    min-height: 44px;
    transition: all 0.2s ease;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: var(--accent) !important;
    background: rgba(250,219,95,0.07) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    border-color: var(--accent) !important;
    background: rgba(250,219,95,0.14) !important;
    box-shadow: 0 0 0 1px rgba(250,219,95,0.28) inset;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child > div {
    border-color: var(--accent) !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
    background-color: var(--accent) !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border-color: var(--bg-border) !important;
    color: #FFFFFF !important;
    font-family: var(--font-mono) !important;
}
::placeholder { color: var(--text-muted) !important; opacity: 1 !important; }

[data-baseweb="select"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"],
[data-testid="stThumbValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploadDropzone"] {
    background-color: var(--bg-panel) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: 10px !important;
    transition: background-color 0.2s;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 180px !important;
    text-align: center !important;
    padding: 2rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(250,219,95,0.05) !important;
}
[data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stFileUploadDropzone"] > div {
    background-color: var(--bg-panel) !important;
}
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploadDropzone"] div,
section[data-testid="stFileUploadDropzone"] span,
section[data-testid="stFileUploadDropzone"] small {
    color: #FFFFFF !important;
    font-weight: 500;
    opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] svg,
section[data-testid="stFileUploadDropzone"] svg {
    fill: var(--accent) !important;
    color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploadDropzone"] button {
    background-color: var(--accent-dim) !important;
    color: #000000 !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    margin: 1rem auto 0.5rem !important;
    display: inline-block !important;
}
[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploadDropzone"] button * {
    color: #000000 !important;
    fill: #000000 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stFileUploadDropzone"] button:hover {
    background-color: var(--accent) !important;
}

[data-testid="stFileUploader"] [data-testid^="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid^="stFileUploadFile"],
[data-testid="stFileUploader"] [data-testid*="FileUploaderFile"],
[data-testid="stFileUploader"] [data-testid*="FileUploadFile"] {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

/* Hide any message below the dropzone */
[data-testid="stFileUploaderDropzone"] ~ div,
section[data-testid="stFileUploadDropzone"] ~ div {
    display: none !important;
}

/* Update the text inside the dropzone */
[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploadDropzone"] small {
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] small::after,
section[data-testid="stFileUploadDropzone"] small::after {
    content: "Max size per file 1 GB • CSV";
    font-size: 0.8rem !important;
    display: inline-block;
    color: var(--text-muted) !important;
}

/* Hide "Press Enter to apply" instruction */
[data-testid="stTextInput"] [data-testid="stWidgetInstructions"],
[data-testid="stTextInput"] .st-emotion-cache-1vt4y6f {
    display: none !important;
}

.brand-logo {
    text-align: center;
    padding: 0.2rem 0 0.1rem;
    font-size: 3rem;
    color: var(--accent);
    font-weight: 700;
}
.brand-sub {
    text-align: center;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.nav-active > button {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(250,219,95,0.08) !important;
    box-shadow: 0 0 0 1px var(--accent) inset !important;
}

.nav-idle > button {
    border-color: var(--bg-border) !important;
}

.nav-active, .nav-idle {
    margin-bottom: -0.85rem !important;
}

[data-testid="stSidebar"] .stButton {
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding: 0.45rem 0.8rem !important;
}
[data-testid="stSidebar"] .stButton > button p {
    white-space: normal !important;
    word-wrap: break-word !important;
    text-align: left !important;
    line-height: 1.25 !important;
    font-size: 0.85rem !important;
}

hr { border-color: var(--bg-border) !important; }

[data-testid="stDataFrameContainer"] {
    border: 1px solid var(--bg-border) !important;
    border-radius: 8px !important;
    background-color: var(--bg-card) !important;
}

/* Ensure the table internals are also dark */
[data-testid="stDataFrameContainer"] [role="gridcell"],
[data-testid="stDataFrameContainer"] [role="columnheader"] {
    background-color: var(--bg-card) !important;
    color: var(--text-main) !important;
}


/* Expander styling */
[data-testid="stExpander"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary {
    background: var(--bg-panel) !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stExpander"] summary p {
    color: var(--accent) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}


[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 1.2rem !important;
    background: var(--bg-card) !important;
    border-radius: 0 0 8px 8px !important;
}
.ai-card-anchor {
    display: none;
}
.ai-card-anchor + div[data-testid="stMarkdownContainer"] {
    background: linear-gradient(145deg, rgba(20, 35, 65, 0.95) 0%, rgba(15, 26, 48, 1) 100%);
    border: 2px solid rgba(250, 219, 95, 0.4) !important;
    border-left: 6px solid var(--accent) !important;
    box-shadow: 0 8px 32px rgba(250, 219, 95, 0.15) !important;
    padding: 2rem !important;
    border-radius: 16px !important;
    margin: 2rem 0 !important;
    animation: insight-glow 3s infinite alternate;
}
@keyframes insight-glow {
    from { box-shadow: 0 8px 32px rgba(250, 219, 95, 0.15); }
    to { box-shadow: 0 8px 48px rgba(250, 219, 95, 0.25); }
}
.ai-card-anchor + div[data-testid="stMarkdownContainer"] h2 {
    color: var(--accent) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 1.5rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.ai-card-anchor + div[data-testid="stMarkdownContainer"] h3 {
    color: var(--accent) !important;
    font-size: 1.3rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid rgba(250, 219, 95, 0.2);
    padding-bottom: 0.5rem;
}
.ai-card-anchor + div[data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.6rem !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}



/* Success Alert Styling */
div[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
}
div[data-testid="stAlert"]:has(svg[aria-label="Success"]) {
    border-color: #00E676 !important;
    background-color: rgba(0, 230, 118, 0.05) !important;
}

/* Copy button & Container */
.copy-container {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(250, 219, 95, 0.04);
    border: 1px solid var(--bg-border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.copy-container:hover {
    border-color: var(--accent);
    background: rgba(250, 219, 95, 0.08);
}
.copy-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #FFFFFF;
    min-width: 100px;
}
.copy-value {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--accent);
    flex-grow: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.copy-btn {
    background: var(--bg-panel) !important;
    border: 1px solid var(--bg-border) !important;
    color: var(--text-muted) !important;
    border-radius: 4px;
    padding: 2px 8px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}
.copy-btn:hover {
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.copy-btn:hover svg {
    fill: var(--accent) !important;
    transform: scale(1.05);
}

/* Example Link Rows */
.example-row {
    display: flex;
    align-items: center;
    background: rgba(250, 219, 95, 0.03) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
    padding: 0.2rem 0.8rem !important;
    margin-bottom: 0.4rem !important;
}
.example-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    min-width: 80px;
}
.example-value {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--accent) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Compact "Use" buttons */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    padding-top: 0.1rem !important;
    padding-bottom: 0.1rem !important;
    font-size: 11px !important;
    height: 30px !important;
    white-space: nowrap !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def apply_layout(fig: Any, title: str = "", height: int = 420) -> Any:
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text=f"<b>{title}</b>",
            x=0.01,
            xanchor="left",
            font=dict(size=15, color=C_PRIMARY),
        ),
        height=height,
    )
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[dict[str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            subtitle_html = f'<div class="kpi-sub">{metric["sub"]}</div>' if metric.get("sub") else ""
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{metric["label"]}</div>'
                f'<div class="kpi-value">{metric["value"]}</div>{subtitle_html}'
                f"</div>",
                unsafe_allow_html=True,
            )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def no_data_gate() -> None:
    st.warning("⚠️  No data loaded. Please return to the **Data Loader** page first.")
    st.stop()


def scroll_to_top() -> None:
    """Inject JS to scroll the main container to the top.

    Uses st.markdown (direct DOM injection) instead of components.html
    (iframe) so the script runs in the main document context with full
    access to scrollable containers.
    """
    st.markdown(
        """
        <script>
            function _scrollToTop() {
                // Target every known Streamlit scrollable wrapper
                var selectors = [
                    '[data-testid="stAppViewContainer"]',
                    '[data-testid="stMain"]',
                    '[data-testid="stMainBlockContainer"]',
                    '.main',
                    '.stApp',
                    'section.main'
                ];
                for (var i = 0; i < selectors.length; i++) {
                    var el = document.querySelector(selectors[i]);
                    if (el) {
                        el.scrollTop = 0;
                        try { el.scrollTo({top: 0, behavior: 'instant'}); } catch(e) {}
                        if (el.parentElement) el.parentElement.scrollTop = 0;
                    }
                }

                // Window / body / html
                window.scrollTo({top: 0, behavior: 'instant'});
                document.body.scrollTop = 0;
                document.documentElement.scrollTop = 0;

                // scrollIntoView on the page-header (always the first rendered element)
                var hdr = document.querySelector('.page-header');
                if (hdr) hdr.scrollIntoView({behavior: 'instant', block: 'start'});
            }

            // Fire immediately + retries to beat Streamlit scroll restoration
            _scrollToTop();
            [30,80,150,250,400,600,900,1300].forEach(function(d){
                setTimeout(_scrollToTop, d);
            });
        </script>
        """,
        unsafe_allow_html=True,
    )


def _render_loaded_dataset_card(df_info: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div style="padding:0.8rem;background:rgba(250,219,95,0.06);border:1px solid #253658;border-radius:8px;font-size:0.78rem;color:#A8B8D0;">
            <div style="color:#FADB5F;font-weight:700;margin-bottom:0.4rem;">📈 Dataset Loaded</div>
            <div>{len(df_info):,} rows &nbsp;·&nbsp; {len(df_info.columns)} cols</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;margin-top:0.3rem;">
                {str(df_info.index.min().date())} → {str(df_info.index.max().date())}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation() -> str:
    with st.sidebar:
        st.markdown('<div class="brand-logo">₿</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Quantitative Analytics</div>', unsafe_allow_html=True)
        st.divider()

        if "page" not in st.session_state:
            st.session_state.page = "Data Loader"

        for icon, name in PAGES:
            css_class = "nav-active" if st.session_state.page == name else "nav-idle"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.page = name
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        if "df" in st.session_state:
            _render_loaded_dataset_card(st.session_state["df"])

    return str(st.session_state.page)
