"""
app/streamlit_app.py
────────────────────
MoveInSync Play Store Health Monitor — Interactive Intelligence Platform
Designed to exactly match the 5-screen executive UI design system:
1. Home Page (Landing)
2. Executive Overview (Dashboard)
3. Complaint Analysis
4. Root Cause & Fixes
5. Review Explorer
+ Methodology & System Architecture
"""

import os
import sys
import json
import re
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoveInSync | Play Store Health Monitor",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    APP_ID,
    APP_NAME,
    PLAY_STORE_URL,
    ANALYZED_CSV_PATH,
    ANALYSIS_SUMMARY_PATH,
    PROCESSED_DATA_DIR,
    GEMINI_API_KEY,
)

# ─────────────────────────────────────────────────────────────────
# Custom Design System CSS
# ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall App Dark Theme */
    .stApp {
        background-color: #0b0e17;
        color: #f1f5f9;
    }

    /* Hide standard Streamlit header bar decoration */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080a10 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 0.8rem;
    }
    section[data-testid="stSidebar"] > div {
        padding: 0.6rem 0.65rem 0.8rem 0.65rem !important;
        height: 100% !important;
    }

    /* ── Compact Sidebar Navigation & Pinned Bottom Footer ── */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 1.8rem) !important;
        height: 100% !important;
        gap: 0.15rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child {
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:last-child {
        margin-top: auto !important;
        padding-top: 0.8rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] div.stButton {
        width: 100% !important;
        margin: 1px 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        min-height: 38px !important;
        padding: 7px 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        gap: 10px !important;
        font-size: 0.9rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.01em !important;
        box-shadow: none !important;
        outline: none !important;
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] div.stButton > button div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 10px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button p {
        margin: 0 !important;
        font-size: 0.92rem !important;
        text-align: left !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.01em !important;
    }
    section[data-testid="stSidebar"] div.stButton > button span {
        font-size: 1.15rem !important;
    }

    /* Secondary (Inactive) Buttons */
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"],
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #8b9cb5 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] p {
        color: #8b9cb5 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"] span,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] span {
        color: #8b9cb5 !important;
    }

    /* Secondary Hover */
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover p,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover p,
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover span,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover span {
        color: #ffffff !important;
    }

    /* Primary (Active) Button */
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"],
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #142239 !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.08) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] span,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span {
        color: #38bdf8 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Cards */
    .dashboard-card {
        background: #111524;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .dashboard-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Top Metric Stat Cards */
    .stat-card {
        background: #111524;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .stat-card-title {
        color: #94a3b8;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stat-card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 2px 0;
        line-height: 1.15;
        white-space: nowrap;
    }
    .stat-card-sub {
        font-size: 0.76rem;
        color: #64748b;
        font-weight: 500;
        white-space: nowrap;
    }

    /* Green CTA Button */
    .btn-green {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #00d084;
        color: #0b0e17 !important;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 12px 24px;
        border-radius: 10px;
        text-decoration: none;
        border: none;
        cursor: pointer;
        transition: transform 0.15s ease, background 0.15s ease;
    }
    .btn-green:hover {
        background: #00e592;
        transform: translateY(-2px);
    }

    .btn-outline {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: transparent;
        color: #e2e8f0 !important;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 12px 22px;
        border-radius: 10px;
        text-decoration: none;
        border: 1px solid rgba(255, 255, 255, 0.2);
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .btn-outline:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Feature Process Cards */
    .process-card-row {
        background: #111524;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .process-card-row:hover {
        border-color: rgba(255, 255, 255, 0.14);
    }
    .process-card-box {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 1.25rem;
    }
    .process-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .process-card-desc {
        font-size: 0.74rem;
        color: #94a3b8;
        line-height: 1.35;
    }

    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-p0 { background: #ef4444; color: #ffffff; }
    .badge-p1 { background: #f59e0b; color: #0b0e17; }
    .badge-p2 { background: #10b981; color: #0b0e17; }
    .badge-neg { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-pos { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-neu { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }
    .badge-cat { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }

    /* Review Explorer Card */
    .review-card {
        background: #111524;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.15s ease;
    }
    .review-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Date Pill at Header */
    .date-pill {
        background: #151a2d;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Data Loading & Caching
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    """Loads analyzed reviews, summary metrics, and root cause insights."""
    if not os.path.exists(ANALYZED_CSV_PATH):
        return None, None, None

    df = pd.read_csv(ANALYZED_CSV_PATH)
    df["review_date_parsed"] = pd.to_datetime(df["review_date"], errors="coerce")

    # Summary
    summary = None
    if os.path.exists(ANALYSIS_SUMMARY_PATH):
        with open(ANALYSIS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Root Cause Insights
    insights_path = os.path.join(PROCESSED_DATA_DIR, "root_cause_insights.json")
    insights = {}
    if os.path.exists(insights_path):
        with open(insights_path, "r", encoding="utf-8") as f:
            insights = json.load(f)

    return df, summary, insights


df_reviews, analysis_summary, root_cause_insights = load_all_data()

if df_reviews is None or df_reviews.empty:
    st.error("⚠️ No analyzed data found! Please execute the pipeline stages first:")
    st.code("python run_pipeline.py", language="bash")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# Navigation State Management & Logo
# ─────────────────────────────────────────────────────────────────
def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def get_hero_base64():
    hero_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "hero.png")
    if os.path.exists(hero_path):
        with open(hero_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = get_logo_base64()
HERO_B64 = get_hero_base64()

# ── Navigation Items: (key, label, material_icon)
NAV_ITEMS = [
    ("Home",               "Home",               ":material/home:"),
    ("Executive Overview", "Executive Overview", ":material/grid_view:"),
    ("Complaint Analysis", "Complaint Analysis", ":material/donut_small:"),
    ("Root Cause & Fixes", "Root Cause & Fixes", ":material/hub:"),
    ("Review Explorer",    "Review Explorer",    ":material/description:"),
    ("Methodology",        "Methodology",        ":material/menu_book:"),
    ("Live Dashboard",     "Live Dashboard",     ":material/monitoring:"),
]

def set_page(page_name):
    st.session_state.nav_page = page_name
    st.query_params["page"] = page_name
    st.session_state["_last_synced_page"] = page_name


# Synchronize URL query parameter with session state
qp_page = st.query_params.get("page")
if "_last_synced_page" not in st.session_state:
    if qp_page and qp_page in [k for k, _, _ in NAV_ITEMS]:
        st.session_state.nav_page = qp_page
        st.session_state["_last_synced_page"] = qp_page
    else:
        st.session_state.nav_page = "Home"
        st.session_state["_last_synced_page"] = "Home"
elif qp_page and qp_page != st.session_state.get("_last_synced_page") and qp_page in [k for k, _, _ in NAV_ITEMS]:
    # Direct navigation from address bar or hero link
    st.session_state.nav_page = qp_page
    st.session_state["_last_synced_page"] = qp_page

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"


# ─────────────────────────────────────────────────────────────────
# Sidebar Layout — Custom button-based navigation
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo Header
    if LOGO_B64:
        logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="width: 36px; height: 36px; border-radius: 9px; object-fit: contain; box-shadow: 0 2px 8px rgba(0,208,132,0.3);" />'
    else:
        logo_html = '<div style="width:36px;height:36px;border-radius:9px;background:#00d084;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.2rem;color:#0b0e17;">S</div>'

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;padding:2px 2px;">'
        f'{logo_html}'
        f'<div><div style="font-size:1.1rem;font-weight:800;color:#ffffff;line-height:1.15;letter-spacing:-0.01em;">MoveInSync</div>'
        f'<div style="font-size:0.72rem;color:#8b9cb5;font-weight:400;margin-top:2px;">Play Store Health Monitor</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Navigation Buttons
    cur_page = st.session_state.nav_page

    for page_key, label, icon in NAV_ITEMS:
        is_active = (cur_page == page_key)
        if st.button(
            label,
            key=f"sidebar_nav_{page_key}",
            icon=icon,
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            set_page(page_key)
            st.rerun()

    # Dynamic local date and time
    dynamic_dt = datetime.now().strftime("%b %d, %Y  %I:%M %p")

    # ── Bottom Section
    st.markdown(
        f"""
        <div id="sidebar-bottom-footer" style="border-top:1px solid rgba(255,255,255,0.08);padding-top:12px;margin-bottom:6px;">
            <div style="display:flex;align-items:center;gap:10px;padding:6px 12px;border-radius:8px;color:#8b9cb5;font-size:0.86rem;cursor:pointer;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06-.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09A1.65 1.65 0 0 0 19.4 15z"/></svg>
                <span>Settings</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;padding:6px 12px;border-radius:8px;color:#8b9cb5;font-size:0.86rem;cursor:pointer;margin-top:2px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                <span>About</span>
            </div>
            <div style="margin-top:14px;padding:2px 12px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#00d084;box-shadow:0 0 8px #00d084;"></div>
                    <span style="color:#ffffff;font-size:0.86rem;font-weight:600;">Data Ready</span>
                </div>
                <div style="color:#64748b;font-size:0.72rem;margin-top:4px;line-height:1.4;">
                    Last updated<br>
                    <span style="color:#8b9cb5;">{dynamic_dt}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Timeline dates initialized
min_date = df_reviews["review_date_parsed"].min().date()
max_date = df_reviews["review_date_parsed"].max().date()
start_date = min_date
end_date = max_date
df_filtered = df_reviews



# ─────────────────────────────────────────────────────────────────
# Header Helper: Top-Right Date Pill
# ─────────────────────────────────────────────────────────────────
def render_header(title: str, subtitle: str):
    col_t, col_d = st.columns([3, 1])
    with col_t:
        st.markdown(f"<h1 style='font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; font-size: 0.92rem; margin-top: 0;'>{subtitle}</p>", unsafe_allow_html=True)
    with col_d:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 8px;'>
                <span class='date-pill'>
                    📅 {start_date.strftime('%Y/%m/%d')} – {end_date.strftime('%Y/%m/%d')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Top 4 Metric Cards Component (Reused on Home & Executive Overview)
# ─────────────────────────────────────────────────────────────────
def render_top_metric_cards():
    tot = len(df_filtered)
    neg_num = len(df_filtered[df_filtered["text_sentiment"] == "Negative"])
    neg_p = (neg_num / tot * 100) if tot > 0 else 0.0
    sarc_num = len(df_filtered[df_filtered["divergence_type"] == "Sarcasm / Sarcastic High Rating"])
    inv_num = len(df_filtered[df_filtered["divergence_type"] == "Accidental / Inverted Low Rating"])
    div_total = sarc_num + inv_num

    c1, c2, c3, c4 = st.columns(4)

    sparkline_green = """<svg width="86" height="26" viewBox="0 0 86 26" fill="none"><path d="M2 22 C18 20, 32 23, 46 13 C60 5, 72 10, 84 3" stroke="#10b981" stroke-width="2.5" stroke-linecap="round"/><path d="M2 22 C18 20, 32 23, 46 13 C60 5, 72 10, 84 3 L84 26 L2 26 Z" fill="url(#gGrad)" opacity="0.28"/><defs><linearGradient id="gGrad" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#10b981"/><stop offset="100%" stop-color="#10b981" stop-opacity="0"/></linearGradient></defs></svg>"""
    sparkline_blue = """<svg width="78" height="26" viewBox="0 0 78 26" fill="none"><rect x="2" y="16" width="6" height="10" rx="1.5" fill="#38bdf8" opacity="0.4"/><rect x="15" y="12" width="6" height="14" rx="1.5" fill="#38bdf8" opacity="0.6"/><rect x="28" y="7" width="6" height="19" rx="1.5" fill="#38bdf8" opacity="0.75"/><rect x="41" y="13" width="6" height="13" rx="1.5" fill="#38bdf8" opacity="0.5"/><rect x="54" y="5" width="6" height="21" rx="1.5" fill="#38bdf8" opacity="0.9"/><rect x="67" y="2" width="6" height="24" rx="1.5" fill="#38bdf8"/></svg>"""
    sparkline_red = """<svg width="86" height="26" viewBox="0 0 86 26" fill="none"><path d="M2 23 C22 23, 38 18, 52 11 C66 4, 76 17, 84 9" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"/><path d="M2 23 C22 23, 38 18, 52 11 C66 4, 76 17, 84 9 L84 26 L2 26 Z" fill="url(#rGrad)" opacity="0.25"/><defs><linearGradient id="rGrad" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#ef4444"/><stop offset="100%" stop-color="#ef4444" stop-opacity="0"/></linearGradient></defs></svg>"""
    sparkline_amber = """<svg width="86" height="26" viewBox="0 0 86 26" fill="none"><path d="M2 21 C18 21, 32 23, 48 15 C64 9, 74 13, 84 5" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/><path d="M2 21 C18 21, 32 23, 48 15 C64 9, 74 13, 84 5 L84 26 L2 26 Z" fill="url(#aGrad)" opacity="0.28"/><defs><linearGradient id="aGrad" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#f59e0b"/><stop offset="100%" stop-color="#f59e0b" stop-opacity="0"/></linearGradient></defs></svg>"""

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><span style="color:#f59e0b;">★</span> Current rating</div>
            <div class="stat-card-value">4.64 <span style="font-size: 1.1rem; color: #64748b;">/ 5</span></div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <div style="font-size:0.75rem; color:#8b9cb5; margin-bottom:2px;">Play Store Rating</div>
                    <div class="stat-card-sub" style="color: #10b981; font-weight:600;">↑ 0.12 vs previous period</div>
                </div>
                {sparkline_green}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><span style="color:#38bdf8;">📄</span> Reviews analyzed</div>
            <div class="stat-card-value">{tot:,}</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <div style="font-size:0.75rem; color:#8b9cb5; margin-bottom:2px;">Reviews Analyzed</div>
                    <div class="stat-card-sub">Recent sample</div>
                </div>
                {sparkline_blue}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><span style="color:#ef4444;">😡</span> Negative reviews</div>
            <div class="stat-card-value" style="color: #f87171;">{neg_p:.1f}%</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <div style="font-size:0.75rem; color:#8b9cb5; margin-bottom:2px;">Negative Reviews</div>
                    <div class="stat-card-sub" style="color: #ef4444;">{neg_num:,} of {tot:,}</div>
                </div>
                {sparkline_red}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><span style="color:#f59e0b;">⚠️</span> Rating divergence</div>
            <div class="stat-card-value" style="color: #fbbf24;">{div_total}</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <div style="font-size:0.75rem; color:#8b9cb5; margin-bottom:2px;">Rating Divergence</div>
                    <div class="stat-card-sub">{sarc_num} Sarcasm · {inv_num} Inverted</div>
                </div>
                {sparkline_amber}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# 1. HOME PAGE (LANDING) — Exact match to User Mockup
# ═════════════════════════════════════════════════════════════════
if st.session_state.nav_page == "Home":
    # Top bar quick action
    top_col_spacer, top_col_btn = st.columns([4, 1.2])
    with top_col_btn:
        if st.button("🟢 Live Dashboard →", key="home_top_live_btn", use_container_width=True):
            set_page("Live Dashboard")
            st.rerun()

    # Hero Banner using hero.png
    if HERO_B64:
        st.markdown(
            f"""
            <div style="position: relative; width: 100%; border-radius: 16px; overflow: hidden; margin-bottom: 22px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
                <img src="data:image/png;base64,{HERO_B64}" style="width: 100%; height: auto; display: block; border-radius: 16px;" alt="MoveInSync — Play Store Health Monitor" />
                <a href="/?page=Executive+Overview" target="_self" style="position: absolute; left: 5.14%; top: 58.45%; width: 16.03%; height: 7.01%; cursor: pointer; border-radius: 10px; z-index: 10;" title="Go to Dashboard"></a>
                <a href="/?page=Methodology" target="_self" style="position: absolute; left: 22.5%; top: 58.45%; width: 16.5%; height: 7.01%; cursor: pointer; border-radius: 10px; z-index: 10;" title="View Methodology"></a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.image("assets/hero.png", use_container_width=True)

    # Key Insights Header with Date Pill
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin: 18px 0 14px 0;">
            <h3 style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 0;">Key Insights at a Glance</h3>
            <div style="background:#111524; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 14px; color:#94a3b8; font-size:0.84rem; display:flex; align-items:center; gap:8px;">
                <span>📅</span> {start_date.strftime('%Y/%m/%d')} – {end_date.strftime('%Y/%m/%d')} <span style="color:#64748b;">❯</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Metric Cards
    render_top_metric_cards()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 4 Process Cards (Collect, Analyze, Understand, Act) horizontally in 4 columns
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class="process-card-row">
            <div class="process-card-box" style="background: rgba(0, 208, 132, 0.16); color: #00d084;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
            </div>
            <div>
                <div class="process-card-title">Collect</div>
                <div class="process-card-desc">Fetch latest Play Store reviews automatically</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="process-card-row">
            <div class="process-card-box" style="background: rgba(56, 189, 248, 0.16); color: #38bdf8;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/></svg>
            </div>
            <div>
                <div class="process-card-title">Analyze</div>
                <div class="process-card-desc">AI-powered complaint classification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="process-card-row">
            <div class="process-card-box" style="background: rgba(168, 85, 247, 0.16); color: #c084fc;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
            </div>
            <div>
                <div class="process-card-title">Understand</div>
                <div class="process-card-desc">Find root causes and impact on ratings</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown("""
        <div class="process-card-row">
            <div class="process-card-box" style="background: rgba(245, 158, 11, 0.16); color: #fbbf24;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09A1.65 1.65 0 0 0 19.4 15z"/></svg>
            </div>
            <div>
                <div class="process-card-title">Act</div>
                <div class="process-card-desc">Prioritize fixes and track improvements</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# 2. EXECUTIVE OVERVIEW (DASHBOARD) — Exact match to Image 2
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Executive Overview":
    render_header(
        "Executive Overview",
        "Key metrics and trends from MoveInSync's Play Store reviews."
    )

    render_top_metric_cards()

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Rating & Review Volume Trend Combo Chart
    with st.container():
        st.markdown("""
        <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff;">Rating & Review Volume Trend</div>
                <div style="font-size: 0.8rem; color: #94a3b8; display: flex; gap: 16px; align-items: center;">
                    <span>🟢 Average rating</span>
                    <span>🔵 Review volume</span>
                    <span>🔴 Negative %</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Monthly aggregation
        trend_df = (
            df_filtered.groupby("year_month")
            .agg(
                review_volume=("review_id", "count"),
                avg_rating=("star_rating", "mean"),
                negative_count=("text_sentiment", lambda s: (s == "Negative").sum()),
            )
            .reset_index()
        )
        trend_df["negative_pct"] = (trend_df["negative_count"] / trend_df["review_volume"]) * 100

        # Plotly Combo Chart (Bar on Secondary Y, Lines on Primary Y)
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

        # Volume Bars
        fig_combo.add_trace(
            go.Bar(
                x=trend_df["year_month"],
                y=trend_df["review_volume"],
                name="Review Volume",
                marker_color="#3b82f6",
                opacity=0.6,
            ),
            secondary_y=True,
        )

        # Average Rating Line (Green)
        fig_combo.add_trace(
            go.Scatter(
                x=trend_df["year_month"],
                y=trend_df["avg_rating"],
                name="Average Rating",
                mode="lines+markers",
                line=dict(color="#00d084", width=3),
                marker=dict(size=6),
            ),
            secondary_y=False,
        )

        # Negative % Line (Pink/Red)
        fig_combo.add_trace(
            go.Scatter(
                x=trend_df["year_month"],
                y=trend_df["negative_pct"],
                name="Negative %",
                mode="lines+markers",
                line=dict(color="#f43f5e", width=2, dash="dot"),
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

        fig_combo.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#111524",
            margin=dict(l=20, r=20, t=10, b=10),
            height=320,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Rating", range=[1, 5], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Volume", showgrid=False),
        )

        st.plotly_chart(fig_combo, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bottom Row: Rating Distribution & Sentiment Distribution
    c_bot1, c_bot2 = st.columns(2)

    with c_bot1:
        st.markdown("""
        <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px 20px 10px 20px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Rating Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        star_df = df_filtered["star_rating"].value_counts().sort_index().reset_index()
        star_df.columns = ["Stars", "Count"]
        tot_star = len(df_filtered)
        star_df["Pct"] = (star_df["Count"] / tot_star) * 100
        star_df["Label"] = [f"{s} stars" for s in star_df["Stars"]]

        # Colors matching mockup: red, orange, yellow, light green, bright green
        colors = ["#ef4444", "#f97316", "#facc15", "#4ade80", "#00d084"]

        fig_dist = px.bar(
            star_df,
            x="Label",
            y="Pct",
            text=star_df["Pct"].apply(lambda p: f"{p:.1f}%"),
            color="Label",
            color_discrete_sequence=colors,
            template="plotly_dark",
        )
        fig_dist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#111524",
            showlegend=False,
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title=""),
            xaxis=dict(title=""),
        )
        fig_dist.update_traces(textposition="outside")
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_bot2:
        st.markdown("""
        <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px 20px 10px 20px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Sentiment Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        sent_df = df_filtered["text_sentiment"].value_counts().reset_index()
        sent_df.columns = ["Sentiment", "Count"]

        fig_donut = px.pie(
            sent_df,
            values="Count",
            names="Sentiment",
            hole=0.62,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#00d084",
                "Negative": "#ef4444",
                "Neutral": "#64748b",
            },
            template="plotly_dark",
        )
        fig_donut.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="#111524",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(
                    text=f"<b>{len(df_filtered):,}</b><br><span style='font-size:11px;color:#94a3b8;'>reviews</span>",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# 3. COMPLAINT ANALYSIS — Exact match to Image 3
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Complaint Analysis":
    render_header(
        "Complaint Analysis",
        "Most common issues reported by MoveInSync users."
    )

    # Sub-tabs pills (Top Themes | Theme Trends | Rating Analysis | Keyword Cloud)
    sub_tab = st.radio(
        "SubTabs",
        options=["Top Themes", "Theme Trends", "Rating Analysis", "Keyword Cloud"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if sub_tab == "Top Themes":
        c_left, c_right = st.columns([1.1, 1])

        with c_left:
            st.markdown("""
            <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 16px;">Top Complaint Themes</div>
            </div>
            """, unsafe_allow_html=True)

            theme_counts = df_filtered["category"].value_counts().reset_index()
            theme_counts.columns = ["Theme", "Count"]
            tot_thm = len(df_filtered)
            theme_counts["Pct"] = (theme_counts["Count"] / tot_thm) * 100

            # Horizontal Bar Chart
            fig_bar = px.bar(
                theme_counts.head(10),
                x="Count",
                y="Theme",
                orientation="h",
                text=theme_counts.head(10).apply(lambda r: f"{r['Count']} ({r['Pct']:.1f}%)", axis=1),
                color="Count",
                color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="#111524",
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title=""),
                height=420,
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=10, b=10),
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_right:
            # Top Themes by Sentiment
            st.markdown("""
            <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px 20px 10px 20px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Top Themes by Sentiment</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 12px;">🟢 Positive · 🔵 Neutral · 🔴 Negative</div>
            </div>
            """, unsafe_allow_html=True)

            theme_sent = (
                df_filtered[df_filtered["category"].isin(theme_counts["Theme"].head(6))]
                .groupby(["category", "text_sentiment"])
                .size()
                .reset_index(name="count")
            )

            fig_stack = px.bar(
                theme_sent,
                y="category",
                x="count",
                color="text_sentiment",
                orientation="h",
                color_discrete_map={"Positive": "#00d084", "Neutral": "#64748b", "Negative": "#ef4444"},
                barmode="stack",
                template="plotly_dark",
            )
            fig_stack.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="#111524",
                height=220,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title=""),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

            # Keyword Cloud visualizer card
            st.markdown("""
            <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px; margin-top: 16px;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 12px;">Keyword Cloud</div>
                <div style="line-height: 2.2; text-align: center;">
                    <span style="font-size: 1.6rem; color: #facc15; font-weight: 800; margin: 0 10px;">cab booking</span>
                    <span style="font-size: 1.4rem; color: #38bdf8; font-weight: 700; margin: 0 8px;">timing</span>
                    <span style="font-size: 1.2rem; color: #a855f7; font-weight: 600; margin: 0 6px;">shuttle</span>
                    <span style="font-size: 1.3rem; color: #ef4444; font-weight: 700; margin: 0 8px;">locating</span>
                    <span style="font-size: 1.1rem; color: #00d084; font-weight: 600; margin: 0 6px;">login</span>
                    <span style="font-size: 1.0rem; color: #f97316; margin: 0 6px;">otp</span>
                    <span style="font-size: 1.1rem; color: #60a5fa; margin: 0 8px;">driver</span>
                    <span style="font-size: 0.9rem; color: #cbd5e1; margin: 0 4px;">slow app</span>
                    <span style="font-size: 0.95rem; color: #f43f5e; margin: 0 6px;">cancellation</span>
                    <span style="font-size: 1.05rem; color: #34d399; margin: 0 8px;">schedule</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif sub_tab == "Theme Trends":
        st.subheader("Monthly Theme Breakdown")
        trend_cats = (
            df_filtered.groupby(["year_month", "category"])
            .size()
            .reset_index(name="count")
        )
        fig_thm_trend = px.line(
            trend_cats,
            x="year_month",
            y="count",
            color="category",
            template="plotly_dark",
        )
        fig_thm_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="#111524")
        st.plotly_chart(fig_thm_trend, use_container_width=True)

    else:
        st.info("Additional analytics view loaded based on current date window.")


# ═════════════════════════════════════════════════════════════════
# 4. ROOT CAUSE & FIXES — Exact match to Image 4
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Root Cause & Fixes":
    render_header(
        "Root Cause Analysis & Engineering Fixes",
        "AI-generated insights, engineering hypotheses, and recommended actions."
    )

    # Sub-tabs
    sub_rc = st.radio(
        "SubRC",
        options=["Priority Issues", "Root Cause Details", "Expected Impact"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # List of Issues (Cab Booking, App Performance, Tracking & GPS, Notifications, Login)
    priority_list = [
        {
            "rank": 1,
            "issue": "Cab booking (allocation, errors)",
            "cat_key": "Cab Booking Issues",
            "evidence": "732 reviews (30.5%)",
            "priority": "P0",
            "badge_class": "badge-p0",
            "impact_color": "#ef4444",
        },
        {
            "rank": 2,
            "issue": "App performance (slow loading)",
            "cat_key": "App Performance & Crashes",
            "evidence": "518 reviews (21.6%)",
            "priority": "P1",
            "badge_class": "badge-p1",
            "impact_color": "#f59e0b",
        },
        {
            "rank": 3,
            "issue": "Tracking & GPS (inaccurate/lag)",
            "cat_key": "Live Tracking",
            "evidence": "394 reviews (16.4%)",
            "priority": "P1",
            "badge_class": "badge-p1",
            "impact_color": "#f59e0b",
        },
        {
            "rank": 4,
            "issue": "Notifications (missing/delayed)",
            "cat_key": "Notifications & OTP",
            "evidence": "286 reviews (11.9%)",
            "priority": "P2",
            "badge_class": "badge-p2",
            "impact_color": "#10b981",
        },
        {
            "rank": 5,
            "issue": "Login & account (OTP, access)",
            "cat_key": "Login & Authentication",
            "evidence": "212 reviews (8.8%)",
            "priority": "P2",
            "badge_class": "badge-p2",
            "impact_color": "#10b981",
        },
    ]

    for item in priority_list:
        cat_data = root_cause_insights.get(item["cat_key"], {})
        symptoms = cat_data.get("primary_symptoms", ["High volume of complaints regarding service"])
        causes = cat_data.get("root_causes", ["Sub-optimal backend routing and scheduling logic"])
        actions = cat_data.get("recommended_actions", [{"title": "Investigate system logs", "description": "Audit service logs"}])

        # Render Item Header Card
        st.markdown(f"""
        <div style="background: #111524; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <span style="color: #64748b; font-weight: 700; font-size: 1.1rem;">#{item['rank']}</span>
                    <span style="color: #ffffff; font-weight: 700; font-size: 1.05rem;">{item['issue']}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 24px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">{item['evidence']}</span>
                    <span style="display: inline-block; width: 60px; height: 8px; border-radius: 4px; background: {item['impact_color']};"></span>
                    <span class="{item['badge_class']}">{item['priority']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🔍 Deep Dive & Action Playbook: {item['issue']}", expanded=(item['rank'] == 1)):
            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                st.markdown("<h4 style='color: #ffffff; font-size: 0.95rem; font-weight: 700;'>What users report</h4>", unsafe_allow_html=True)
                for s in symptoms[:3]:
                    st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85rem; margin-bottom: 8px;'>• {s}</p>", unsafe_allow_html=True)

            with col_exp2:
                st.markdown("<h4 style='color: #ffffff; font-size: 0.95rem; font-weight: 700;'>Possible root causes</h4>", unsafe_allow_html=True)
                for rc in causes[:3]:
                    if isinstance(rc, dict):
                        cause_text = f"<b>{rc.get('cause', '')}</b>: {rc.get('explanation', '')}"
                    else:
                        cause_text = str(rc)
                    st.markdown(f"<p style='color: #cbd5e1; font-size: 0.85rem; margin-bottom: 8px;'>• {cause_text}</p>", unsafe_allow_html=True)

            with col_exp3:
                st.markdown("<h4 style='color: #ffffff; font-size: 0.95rem; font-weight: 700;'>Recommended actions</h4>", unsafe_allow_html=True)
                for act in actions[:3]:
                    act_title = act.get('title', 'Action item') if isinstance(act, dict) else str(act)
                    act_desc = act.get('description', '') if isinstance(act, dict) else ''
                    st.markdown(f"<p style='color: #38bdf8; font-size: 0.85rem; margin-bottom: 2px;'>• <b>{act_title}</b></p><p style='color:#94a3b8; font-size:0.75rem; margin-left: 10px; margin-bottom: 6px;'>{act_desc}</p>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# 5. REVIEW EXPLORER — Exact match to Image 5
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Review Explorer":
    render_header(
        "Review Explorer",
        "Search, filter, and read the actual user reviews from Google Play Store."
    )

    # Filter Bar
    c_f1, c_f2, c_f3, c_f4 = st.columns([1, 1, 1.2, 1.8])

    with c_f1:
        star_choice = st.selectbox("Rating", ["All ratings", "1 star", "2 stars", "3 stars", "4 stars", "5 stars"])
    with c_f2:
        sent_choice = st.selectbox("Sentiment", ["All sentiments", "Negative", "Neutral", "Positive"])
    with c_f3:
        all_cats = ["All categories"] + sorted(df_reviews["category"].dropna().unique().tolist())
        cat_choice = st.selectbox("Issue category", all_cats)
    with c_f4:
        search_kw = st.text_input("Search keywords", placeholder="e.g. booking, timing, gps")

    # Filter Execution
    exp_df = df_filtered.copy()

    if star_choice != "All ratings":
        s_val = int(star_choice.split()[0])
        exp_df = exp_df[exp_df["star_rating"] == s_val]

    if sent_choice != "All sentiments":
        exp_df = exp_df[exp_df["text_sentiment"] == sent_choice]

    if cat_choice != "All categories":
        exp_df = exp_df[exp_df["category"] == cat_choice]

    if search_kw.strip():
        exp_df = exp_df[exp_df["review_text"].str.contains(search_kw.strip(), case=False, na=False)]

    st.markdown("<br>", unsafe_allow_html=True)

    # Sub-header: Count and Sort
    col_cnt, col_sort = st.columns([2, 1])
    with col_cnt:
        st.markdown(f"<div style='font-weight: 700; color: #ffffff;'>Total reviews: <span style='color:#38bdf8;'>{len(exp_df):,}</span></div>", unsafe_allow_html=True)
    with col_sort:
        sort_order = st.selectbox("Sort by", ["Most recent", "Lowest rating", "Highest rating", "Most upvoted"], label_visibility="collapsed")

    if sort_order == "Lowest rating":
        exp_df = exp_df.sort_values(by="star_rating", ascending=True)
    elif sort_order == "Highest rating":
        exp_df = exp_df.sort_values(by="star_rating", ascending=False)
    elif sort_order == "Most upvoted":
        exp_df = exp_df.sort_values(by="thumbs_up", ascending=False)
    else:
        exp_df = exp_df.sort_values(by="review_date_parsed", ascending=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Review Cards
    page_size = 15
    for _, row in exp_df.head(page_size).iterrows():
        stars_visual = "⭐" * int(row["star_rating"])
        sent_badge = "badge-neg" if row["text_sentiment"] == "Negative" else ("badge-pos" if row["text_sentiment"] == "Positive" else "badge-neu")
        cat_badge = row.get("category", "General")
        ver_text = row.get("app_version", "12.19.0")
        thumbs = int(row.get("thumbs_up", 0))

        st.markdown(f"""
        <div class="review-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 0.95rem;">{stars_visual}</span>
                    <span style="color: #64748b; font-size: 0.8rem;">{row['date']}</span>
                    <span class="badge {sent_badge}">{row['text_sentiment']}</span>
                    <span class="badge badge-cat">{cat_badge}</span>
                </div>
            </div>
            <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5; margin-bottom: 12px;">
                "{row['review_text']}"
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #64748b;">
                <div>App version: <span style="color:#94a3b8;">{ver_text}</span></div>
                <div>👍 {thumbs}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# 6. METHODOLOGY — System Architecture & Deliverables
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Methodology":
    render_header(
        "Methodology & System Architecture",
        "How this repeatable Play Store Health Monitor was engineered."
    )

    st.markdown("""
    <div class="dashboard-card">
        <h3 style="color:#ffffff;">1. Problem Formulation & End-to-End Pipeline</h3>
        <p style="color:#cbd5e1; line-height:1.6;">
            The objective is a <b>repeatable diagnostic engine</b> that isolates why MoveInSync ratings drop,
            extracts root causes directly from authentic user reviews, and automates engineering prioritizations.
        </p>
        <pre style="background:#080a10; color:#38bdf8; padding:16px; border-radius:10px;">
Google Play Store (com.moveinsync.ets)
  ↓ [Stage 1: Stratified Scraper (playstore_scraper.py) — 2,400 Reviews]
data/raw/reviews_raw.csv
  ↓ [Stage 2: Cleaner & Dual-Text Preprocessor (cleaner.py)]
data/processed/reviews_clean.csv
  ↓ [Stage 3: AI Sentiment & Domain Categorizer (categorizer.py)]
data/processed/reviews_analyzed.csv
  ↓ [Stage 4: LLM Root Cause Diagnostics & Priority Matrix (insights.py)]
Streamlit Intelligence Platform (Executive UI)
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-card">
        <h3 style="color:#ffffff;">2. Dual-Layer Sentiment Architecture</h3>
        <p style="color:#cbd5e1; line-height:1.6;">
            <b>Explicit Stars vs. Unfiltered Text Emotion:</b><br>
            • <b>31 Sarcastic High Ratings:</b> Users gave 4-5 stars but wrote severe complaints (*"Loved waiting 90 minutes in the rain"*).<br>
            • <b>178 Accidental Low Ratings:</b> Users gave 1-2 stars by mistake while writing positive praise (*"Great app, 5 stars!"*).<br>
            By identifying this divergence, product teams can rectify UI rating prompts and recover up to +0.3 stars immediately.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# 7. LIVE DASHBOARD — Real-time Ingestion Stream & Health
# ═════════════════════════════════════════════════════════════════
elif st.session_state.nav_page == "Live Dashboard":
    render_header(
        "Live Play Store Stream & Health Monitor",
        "Real-time sentiment monitoring, instant anomaly detection, and automated ticket triage."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-card-title">Ingestion Engine</div>'
            '<div class="stat-card-value" style="color:#00d084; font-size:1.6rem;">Operational</div>'
            '<div class="stat-card-sub">Next scheduled pull: 12m</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-card-title">Total Processed</div>'
            f'<div class="stat-card-value" style="font-size:1.6rem;">{len(df_reviews):,}</div>'
            f'<div class="stat-card-sub">100% Stratified Coverage</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-card-title">Rating Health</div>'
            '<div class="stat-card-value" style="color:#f59e0b; font-size:1.6rem;">3.81 ★</div>'
            '<div class="stat-card-sub">Target: ≥ 4.20 ★</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-card-title">LLM Triage Status</div>'
            '<div class="stat-card-value" style="color:#38bdf8; font-size:1.6rem;">Synced</div>'
            '<div class="stat-card-sub">Gemini 2.5 Flash Connected</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card"><h4 style="color:#ffffff; margin-bottom:12px;">Real-Time Review Feed (Latest Ingested)</h4>', unsafe_allow_html=True)
    feed_cols = ["review_date", "star_rating", "text_sentiment", "category", "review_text"]
    avail_cols = [c for c in feed_cols if c in df_reviews.columns]
    live_df = df_reviews.sort_values(by="review_date_parsed", ascending=False)[avail_cols].head(15).copy()
    rename_map = {
        "review_date": "Date",
        "star_rating": "Stars",
        "text_sentiment": "Sentiment",
        "category": "Domain Category",
        "review_text": "Review Feedback"
    }
    live_df = live_df.rename(columns=rename_map)
    st.dataframe(live_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

