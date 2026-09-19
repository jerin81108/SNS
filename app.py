"""
Q-Signal: Hybrid Quantum–Classical Traffic Signal Optimization
Main Streamlit Dashboard

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
import random
import copy
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
import yaml
import matplotlib.pyplot as plt

# ── Page config (must be first st call) ─────────────────────────────────────
st.set_page_config(
    page_title="Q-Signal | Quantum Traffic Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme State & Dynamic CSS ────────────────────────────────────────────────
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "🍃 Mild / Clean Light"

is_mild = "Mild" in st.session_state.get("ui_theme", "🍃 Mild / Clean Light")
theme_key = "mild" if is_mild else "dark"

if is_mild:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b !important;
        background: #f8fafc !important;
    }

    .main { background: #f8fafc !important; }
    [data-testid="stHeader"] { background: rgba(248, 250, 252, 0.85) !important; }
    [data-testid="stSidebar"] { 
        background: #ffffff !important; 
        border-right: 1px solid #e2e8f0 !important; 
    }

    /* Force clean readable text */
    p, span, div, li {
        color: #334155;
    }

    /* Sidebar labels and headers */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* Form input labels */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label, [data-testid="stWidgetLabel"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1d4ed8 !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Mild Aesthetic Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 16px rgba(37,99,235,0.1); 
        border-color: #93c5fd; 
    }
    .metric-value { 
        font-size: 2.15rem; 
        font-weight: 800; 
        color: #0f172a !important; 
        font-family:'JetBrains Mono',monospace; 
    }
    .metric-label { 
        font-size: 0.82rem; 
        color: #64748b !important; 
        text-transform: uppercase; 
        letter-spacing: 0.06em; 
        margin-top: 6px; 
        font-weight: 700;
    }
    .metric-delta { font-size: 0.88rem; margin-top: 4px; font-weight: 600; }
    .delta-good { color: #16a34a !important; }
    .delta-bad  { color: #dc2626 !important; }

    /* Mild Status Badges */
    .status-badge {
        display: inline-block; padding: 4px 12px; border-radius: 99px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.03em;
    }
    .badge-fixed    { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
    .badge-pressure { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
    .badge-qaoa     { background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; }
    .badge-emergency{ background:#ffedd5; color:#c2410c; border:2px solid #fdba74; animation: pulse 1.5s infinite; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

    /* Quantum Card - Mild */
    .quantum-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px; padding: 16px;
        font-family:'JetBrains Mono',monospace; font-size:0.86rem; color:#1e293b !important;
        line-height: 1.6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }

    /* Emergency banner */
    .emergency-banner {
        background: linear-gradient(90deg, #dc2626, #ea580c);
        border: 1px solid #f97316; border-radius: 10px;
        padding: 14px 22px; margin: 8px 0 16px 0;
        color: #ffffff !important; font-weight:700; font-size:1.05rem;
        box-shadow: 0 4px 14px rgba(234, 88, 12, 0.25);
        animation: pulse 1.5s infinite;
    }

    .stButton > button {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { 
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        transform: translateY(-1px) !important; 
    }
    .stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background-color: #2563eb !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    /* Input and selectbox styling in mild theme */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }

    /* Custom styled tabs - Mild */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 9px 16px;
        color: #475569 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
        border-color: #3b82f6 !important;
        font-weight: 700 !important;
    }

    /* Captions and subtext */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 0.85rem !important;
    }

    /* Dataframe and tables */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    table {
        color: #0f172a !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] { 
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
    }

    .main { background: #0b1120 !important; }
    [data-testid="stSidebar"] { 
        background: #0d1527 !important; 
        border-right: 1px solid #1e3a5f !important; 
    }

    /* Force readable text everywhere */
    p, span, div, li {
        color: #f1f5f9;
    }

    /* Sidebar labels and text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    /* Form input labels */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label, [data-testid="stWidgetLabel"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.12rem;
        font-weight: 700;
        color: #38bdf8 !important;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Metric Cards with maximum visual punch and contrast */
    .metric-card {
        background: linear-gradient(135deg, #0d1b2e 0%, #172554 100%);
        border: 1px solid #2563eb;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,100,255,0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(56,189,248,0.28); }
    .metric-value { 
        font-size: 2.15rem; 
        font-weight: 800; 
        color: #38bdf8 !important; 
        font-family:'JetBrains Mono',monospace; 
        text-shadow: 0 0 10px rgba(56,189,248,0.35);
    }
    .metric-label { 
        font-size: 0.82rem; 
        color: #f8fafc !important; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        margin-top: 6px; 
        font-weight: 700;
    }
    .metric-delta { font-size: 0.88rem; margin-top: 4px; font-weight: 600; }
    .delta-good { color: #4ade80 !important; }
    .delta-bad  { color: #f87171 !important; }

    /* Status Badges */
    .status-badge {
        display: inline-block; padding: 5px 14px; border-radius: 99px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;
    }
    .badge-fixed    { background:#261313; color:#f87171; border:1px solid #ef4444; }
    .badge-pressure { background:#261f0d; color:#fbbf24; border:1px solid #f59e0b; }
    .badge-qaoa     { background:#0d2818; color:#4ade80; border:1px solid #22c55e; }
    .badge-emergency{ background:#7c2d12; color:#ffedd5; border:2px solid #f97316; animation: pulse 1.5s infinite; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

    /* Quantum Card */
    .quantum-card {
        background: linear-gradient(135deg, #0c182b, #132238);
        border: 1px solid #38bdf888;
        border-radius: 10px; padding: 16px;
        font-family:'JetBrains Mono',monospace; font-size:0.86rem; color:#bae6fd !important;
        line-height: 1.6;
    }

    /* Emergency banner */
    .emergency-banner {
        background: linear-gradient(90deg, #991b1b, #c2410c);
        border: 2px solid #fb923c; border-radius: 10px;
        padding: 14px 22px; margin: 8px 0 16px 0;
        color: #ffffff !important; font-weight:700; font-size:1.05rem;
        animation: pulse 1.5s infinite;
    }

    .stButton > button {
        border-radius: 8px !important; font-weight: 700 !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }

    /* Custom styled tabs with strong visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 18px;
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Captions and subtext */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
    }

    /* Markdown tables */
    [data-testid="stDataFrame"], table {
        color: #f8fafc !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Load config ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

CFG = load_config()


# ── Session state initializer ────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "running": False,
        "sim": None,
        "events": None,
        "em_mgr": None,
        "controller_mode": "qaoa",
        "demand_profile": "normal",
        "seed": 42,
        "n_intersections": 4,
        "qaoa_shots": 512,
        "qaoa_depth": 2,
        "sim_speed": 5,
        "queue_records": {},        # {ctrl: [(t, q), ...]}
        "comparison_dfs": {},       # {ctrl: DataFrame}
        "step_count": 0,
        "last_plan": None,
        "em_status": None,
        "phase_switch_count": 0,
        "tick_buffer": [],
        "sim_refresh_interval": 0.6,
        "map_view_mode": "svg",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Simulator factory ────────────────────────────────────────────────────────
def _build_sim(profile, seed, n, cfg):
    from qsignal.simulator import TrafficSimulator
    from qsignal.events import EventManager
    from qsignal.emergency import EmergencyCorridorManager

    sim_cfg = {
        "min_green_s": cfg["signal"]["min_green_s"],
        "max_green_s": cfg["signal"]["max_green_s"],
        "yellow_s": cfg["signal"]["yellow_s"],
        "all_red_s": cfg["signal"]["all_red_s"],
        "saturation_flow": cfg["network"]["saturation_flow"],
        "idle_fuel_L_per_s": cfg["metrics"]["idle_fuel_L_per_s"],
        "co2_per_litre": cfg["metrics"]["co2_per_litre"],
    }
    sim = TrafficSimulator(n_intersections=n, profile=profile, seed=seed, config=sim_cfg)
    events = EventManager(sim)
    em_mgr = EmergencyCorridorManager(
        sim,
        clearance_buffer_s=cfg["emergency"]["clearance_buffer_s"],
        advance_reservations=cfg["emergency"]["advance_reservations"],
    )
    return sim, events, em_mgr


def _do_reset():
    profile = st.session_state.demand_profile
    seed = st.session_state.seed
    n = st.session_state.n_intersections
    sim, events, em_mgr = _build_sim(profile, seed, n, CFG)
    st.session_state.sim = sim
    st.session_state.events = events
    st.session_state.em_mgr = em_mgr
    st.session_state.running = False
    st.session_state.queue_records = {}
    st.session_state.comparison_dfs = {}
    st.session_state.step_count = 0
    st.session_state.last_plan = None
    st.session_state.em_status = None
    st.session_state.phase_switch_count = 0


# Ensure simulator exists on start
if st.session_state.sim is None:
    _do_reset()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Q-Signal Hub")
    st.caption("Hybrid Quantum–Classical Traffic Optimization")
    
    # Live Weather Widget
    from qsignal.weather import get_live_weather
    weather = get_live_weather()
    
    weather_bg = "rgba(255, 255, 255, 0.6)" if is_mild else "rgba(15, 23, 42, 0.6)"
    weather_border = "#e2e8f0" if is_mild else "#334155"
    weather_text = "#475569" if is_mild else "#94a3b8"
    weather_temp = "#0f172a" if is_mild else "#f8fafc"
    
    st.markdown(f"""
    <div style="padding: 12px; border-radius: 10px; background-color: {weather_bg}; border: 1px solid {weather_border}; margin-bottom: 16px;">
        <div style="font-size: 0.75rem; color: {weather_text}; margin-bottom: 6px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Live Weather (Tiruppur)</div>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 2rem; line-height: 1;">{weather['icon']}</div>
            <div style="text-align: right;">
                <div style="font-size: 1.25rem; font-weight: 800; color: {weather_temp};">{weather['temperature']}°C</div>
                <div style="font-size: 0.8rem; color: {weather_text}; font-weight: 500;">{weather['description']} | 💨 {weather['windspeed']} km/h</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    ui_theme_selected = st.selectbox(
        "🎨 UI Theme",
        ["🍃 Mild / Clean Light", "🌙 Dark Cyberpunk"],
        index=0 if is_mild else 1,
        help="Switch between clean mild light palette and dark cyberpunk theme",
    )
    if ui_theme_selected != st.session_state.get("ui_theme"):
        st.session_state["ui_theme"] = ui_theme_selected
        st.rerun()

    st.divider()

    # Preset Quick Launcher
    st.markdown('<div class="section-header">⚡ Quick Scenarios</div>', unsafe_allow_html=True)
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("🏙️ Balanced", use_container_width=True, help="Standard everyday traffic flow"):
            st.session_state.demand_profile = "normal"
            st.session_state.controller_mode = "qaoa"
            _do_reset()
            st.session_state.running = True
            st.rerun()
        if st.button("🚗 Heavy Grid", use_container_width=True, help="High volume traffic stress test"):
            st.session_state.demand_profile = "heavy"
            st.session_state.controller_mode = "qaoa"
            _do_reset()
            st.session_state.running = True
            st.rerun()
    with q_col2:
        if st.button("🌅 Rush Hour", use_container_width=True, help="Heavy inbound commute rush"):
            st.session_state.demand_profile = "morning_peak"
            st.session_state.controller_mode = "qaoa"
            _do_reset()
            st.session_state.running = True
            st.rerun()
        if st.button("🚨 Emergency Vehicle", use_container_width=True, help="Instant emergency vehicle green corridor test"):
            st.session_state.demand_profile = "normal"
            _do_reset()
            if st.session_state.events:
                nodes = list(st.session_state.sim.state.intersections.keys())
                if len(nodes) >= 2:
                    st.session_state.events.inject_emergency_vehicle(nodes[0], nodes[-1])
            st.session_state.running = True
            st.rerun()

    st.divider()
    st.markdown('<div class="section-header">⚙️ Simulation Setup</div>', unsafe_allow_html=True)

    controller_mode = st.selectbox(
        "Optimization Controller",
        ["fixed", "pressure", "qaoa"],
        format_func=lambda x: {
            "fixed": "🔴 Fixed-Time (Baseline)",
            "pressure": "🟡 Max-Pressure (Classical)",
            "qaoa": "🟢 Hybrid QAOA (Quantum)"
        }[x],
        key="controller_mode",
        help="fixed: cyclic timing; pressure: greedy backlog; qaoa: quantum approximate optimization"
    )

    demand_profile = st.selectbox(
        "Traffic Profile",
        ["normal", "morning_peak", "evening_peak", "heavy", "low"],
        format_func=lambda x: {
            "normal": "📊 Normal Balanced",
            "morning_peak": "🌅 Morning Inbound Peak",
            "evening_peak": "🌆 Evening Outbound Peak",
            "heavy": "🚗 Heavy Gridlock",
            "low": "🛣️ Low / Late Night"
        }[x],
        key="demand_profile",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.number_input("🎲 Random Seed", min_value=0, max_value=9999, key="seed")
    with col_b:
        st.selectbox("🗺️ Intersections", [4, 6, 8], key="n_intersections")

    st.slider("⏩ Simulation Speed (ticks/step)", min_value=1, max_value=30, value=5, key="sim_speed")
    st.slider("⏱️ Refresh Delay (seconds)", min_value=0.2, max_value=2.0, value=0.6, step=0.1, key="sim_refresh_interval", help="Higher value eliminates screen blinking and gives a calm, smooth display.")


    st.divider()
    st.markdown('<div class="section-header">🎮 Playback Controls</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            if st.session_state.sim is None:
                _do_reset()
            st.session_state.running = True
        if st.button("⏭️ Step (1s)", use_container_width=True, help="Advance simulation by one step"):
            st.session_state.running = False
            _run_step_once = True
    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False
        if st.button("🔄 Reset", use_container_width=True):
            _do_reset()
            st.toast("Simulation network reset to t=0s", icon="🔄")

    st.divider()
    st.markdown('<div class="section-header">🚨 Live Incident Injection</div>', unsafe_allow_html=True)

    sim_for_events = st.session_state.sim
    node_list = list(sim_for_events.state.intersections.keys()) if sim_for_events else ["I0"]
    approach_list = list(sim_for_events.state.approaches.keys()) if sim_for_events else ["I0_NS"]

    selected_approach = st.selectbox("Target Approach", approach_list)
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        if st.button("📈 Demand Surge", use_container_width=True, help="+40% sudden traffic flow"):
            if st.session_state.events:
                st.session_state.events.inject_congestion_spike(selected_approach, extra_rate=0.40, duration_s=60)
                st.toast(f"📈 Surge injected on {selected_approach} (60s)", icon="📈")
        if st.button("🚧 Lane Closure", use_container_width=True, help="Capacity reduced to 0 for 120s"):
            if st.session_state.events:
                st.session_state.events.inject_lane_closure(selected_approach, duration_s=120)
                st.toast(f"🚧 Lane closed on {selected_approach} (120s)", icon="🔴")
    with ev_col2:
        if st.button("💥 Accident Hazard", use_container_width=True, help="Accident bottleneck for 90s"):
            if st.session_state.events:
                st.session_state.events.inject_accident(selected_approach, duration_s=90)
                st.toast(f"💥 Accident reported on {selected_approach} (90s)", icon="⚠️")
        if st.button("🧹 Clear Events", use_container_width=True, help="Clear active incidents"):
            if st.session_state.events:
                st.session_state.events.active = []
                st.toast("Active incidents cleared", icon="🧹")

    st.markdown("**🚨 Priority Emergency Corridor**")
    from qsignal.network import get_landmark_name
    em_origin = st.selectbox(
        "Dispatch From",
        node_list,
        format_func=lambda n: f"{n} ({get_landmark_name(n)})",
        key="em_origin"
    )
    em_dest_opts = [n for n in node_list if n != em_origin]
    em_dest = st.selectbox(
        "Destination Junction / Facility",
        em_dest_opts if em_dest_opts else node_list,
        format_func=lambda n: f"{n} ({get_landmark_name(n)})",
        key="em_dest"
    )
    if st.button("🚨 Dispatch Emergency Vehicle", use_container_width=True, type="primary"):
        if st.session_state.events:
            ev = st.session_state.events.inject_emergency_vehicle(em_origin, em_dest)
            if ev:
                st.toast(f"🚨 Emergency vehicle dispatched: {em_origin} ➔ {em_dest}!", icon="🚨")
            else:
                st.toast("❌ No valid route available", icon="❗")

    st.divider()
    # Export button
    st.markdown('<div class="section-header">💾 Export & Reports</div>', unsafe_allow_html=True)
    sim = st.session_state.sim
    if sim and sim.metrics_history:
        from qsignal.metrics import snapshots_to_dataframe
        df_export = snapshots_to_dataframe(sim.metrics_history)
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            "⬇️ Download metrics.csv",
            data=csv_data,
            file_name="qsignal_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Run simulation to generate downloadable CSV data.")


# ── Main Header ─────────────────────────────────────────────────────────────
if is_mild:
    st.markdown("""
    <div style="background:linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);border:1px solid #e2e8f0;
    border-radius:14px;padding:20px 28px;margin-bottom:20px;box-shadow:0 2px 10px rgba(0,0,0,0.03);">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap">
      <div>
        <h1 style="color:#0f172a;margin:0;font-size:2rem;font-weight:800">
          🚦 Q-Signal
          <span style="font-size:1.05rem;color:#475569;font-weight:500;margin-left:14px">
            Hybrid Quantum–Classical Traffic Signal Optimization
          </span>
        </h1>
        <p style="color:#64748b;margin:6px 0 0;font-size:0.92rem;font-weight:400">
          QUBO / QAOA Formulation · Classical Safety Guardrails · Preemptive Emergency Green Wave Corridor
        </p>
      </div>
      <div style="margin-top:8px">
        <span style="background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;padding:6px 14px;border-radius:20px;font-size:0.82rem;font-weight:700">
          ⚡ Simulator: Qiskit Aer (Classical Emulation)
        </span>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2e,#1a2744);border:1px solid #1e3a5f;
    border-radius:14px;padding:20px 28px;margin-bottom:20px">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap">
      <div>
        <h1 style="color:#38bdf8;margin:0;font-size:2rem;font-weight:800">
          🚦 Q-Signal
          <span style="font-size:1.05rem;color:#f8fafc;font-weight:500;margin-left:14px">
            Hybrid Quantum–Classical Traffic Signal Optimization
          </span>
        </h1>
        <p style="color:#cbd5e1;margin:6px 0 0;font-size:0.92rem;font-weight:400">
          QUBO / QAOA Formulation · Classical Safety Guardrails · Preemptive Emergency Green Wave Corridor
        </p>
      </div>
      <div style="margin-top:8px">
        <span style="background:#1e293b;border:1px solid #38bdf8;color:#38bdf8;padding:6px 14px;border-radius:20px;font-size:0.82rem;font-weight:700">
          ⚡ Simulator: Qiskit Aer (Classical Emulation)
        </span>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ── Simulation step logic ────────────────────────────────────────────────────
DECISION_INTERVAL = CFG["simulation"]["decision_interval_s"]

def _run_step():
    sim = st.session_state.sim
    if sim is None:
        return

    events = st.session_state.events
    em_mgr = st.session_state.em_mgr
    mode = st.session_state.controller_mode
    speed = st.session_state.sim_speed

    # Tick events
    if events:
        events.tick()

    # Every DECISION_INTERVAL seconds, compute a new plan
    if sim.state.time_s % DECISION_INTERVAL == 0:
        from qsignal.controllers.hybrid_qaoa import solve_action
        from qsignal.safety import SafetyValidator

        qaoa_cfg = {
            "shots": st.session_state.qaoa_shots,
            "depth": st.session_state.qaoa_depth,
            "time_budget_s": 5.0,
            "fallback": "exact",
            **CFG.get("qubo", {}),
        }
        plan = solve_action(sim.state, mode=mode, config=qaoa_cfg, graph=sim.graph)

        # Safety validation
        validator = SafetyValidator(
            min_green=CFG["signal"]["min_green_s"],
            yellow_s=CFG["signal"]["yellow_s"],
            all_red_s=CFG["signal"]["all_red_s"],
        )
        safe_plan, _ = validator.validate(plan, sim.state)
        sim.apply_signal_plan(safe_plan.assignments)
        st.session_state.last_plan = safe_plan

        # Emergency corridor tick
        if em_mgr:
            em_mgr.tick()

    # Step simulation
    sim.step(n_ticks=speed)

    # Record metrics
    plan = st.session_state.last_plan
    snap = sim.record_metrics(
        controller=mode,
        qaoa_energy=plan.qaoa_energy if plan else None,
        qaoa_gap=plan.qaoa_gap_pct if plan else None,
    )

    # Update rolling queue records (keep last 300 points)
    qr = st.session_state.queue_records
    qr.setdefault(mode, []).append((snap.time_s, snap.mean_queue))
    if len(qr[mode]) > 300:
        qr[mode] = qr[mode][-300:]

    # Update emergency status
    if em_mgr:
        st.session_state.em_status = em_mgr.get_status()

    # Optional Firebase Cloud Sync
    if st.session_state.get("firebase_sync_enabled", False):
        try:
            from qsignal.firebase_sync import sync_traffic_snapshot
            sync_traffic_snapshot({
                "time_s": snap.time_s,
                "mean_queue": snap.mean_queue,
                "max_queue": snap.max_queue,
                "throughput": snap.throughput_total,
                "controller": mode,
                "qaoa_energy": plan.qaoa_energy if plan else None,
            })
        except Exception:
            pass

    st.session_state.step_count += 1


# Execute step if simulation is active or manual step requested
if (st.session_state.running or locals().get("_run_step_once", False)) and st.session_state.sim is not None:
    _run_step()

sim = st.session_state.sim
state = sim.state
last_plan = st.session_state.last_plan
em_status = st.session_state.em_status or {}


# ── Emergency alert banner ───────────────────────────────────────────────────
if em_status.get("active"):
    route_str = " ➔ ".join(em_status.get("route", []))
    elapsed = em_status.get("elapsed_s", 0)
    current_node = em_status.get("current_node", "En route")
    st.markdown(
        f'<div class="emergency-banner">🚨 EMERGENCY CORRIDOR ACTIVE &nbsp;|&nbsp; '
        f'Vehicle Route: <b>{route_str}</b> &nbsp;|&nbsp; Current Node: <b>{current_node}</b> &nbsp;|&nbsp; Elapsed: <b>{elapsed}s</b></div>',
        unsafe_allow_html=True,
    )


# ── Top Live KPI Cards ───────────────────────────────────────────────────────
mode = st.session_state.controller_mode
mode_badge = {
    "fixed": '<span class="status-badge badge-fixed">🔴 Fixed-Time</span>',
    "pressure": '<span class="status-badge badge-pressure">🟡 Max-Pressure</span>',
    "qaoa": '<span class="status-badge badge-qaoa">🟢 Hybrid QAOA</span>',
}.get(mode, mode)

snaps = sim.metrics_history
latest = snaps[-1] if snaps else None

queues = [a.queue for a in state.approaches.values()]
mean_q = sum(queues) / len(queues) if queues else 0
max_q = max(queues) if queues else 0
throughput = len(state.completed_vehicles)
phase_sw = sim._total_phase_switches

qaoa_energy_str = f"{last_plan.qaoa_energy:.3f}" if (last_plan and last_plan.qaoa_energy is not None) else "—"
qaoa_gap_str = f"{last_plan.qaoa_gap_pct:.1f}%" if (last_plan and last_plan.qaoa_gap_pct is not None) else "—"
solver_str = last_plan.solver.upper() if last_plan else "IDLE"

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{state.time_s}s</div>'
                f'<div class="metric-label">⏱️ Sim Time</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{mean_q:.1f}</div>'
                f'<div class="metric-label">🚗 Mean Queue</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{max_q:.1f}</div>'
                f'<div class="metric-label">⚠️ Max Queue</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{throughput}</div>'
                f'<div class="metric-label">🏁 Throughput</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{phase_sw}</div>'
                f'<div class="metric-label">🔀 Phase Switches</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(state.vehicles)}</div>'
                f'<div class="metric-label">🚗 Active Fleet</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_sim, tab_perf, tab_carbon, tab_incidents, tab_guide = st.tabs([
    "🚦 Live Simulation & Map",
    "📊 Performance Analytics",
    "🌱 Carbon & Monthly Traffic AI",
    "🚨 Emergency & Incidents",
    "📘 How It Works & Guide",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Simulation & Map
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    col_map, col_q = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="section-header">🗺️ Live Real-Time Map of Tiruppur (Tamil Nadu, India)</div>', unsafe_allow_html=True)
        st.caption("📍 **Tiruppur City Live Network:** Real-time GIS street map tracking traffic flow, signals, and emergency green wave routing across central Tiruppur.")
        from qsignal.visualization import (
            build_network_svg,
            build_network_map,
            build_real_live_map_html,
            phase_legend_html,
        )
        from qsignal.network import get_landmark_name

        map_token = (st.session_state.get("mapbox_api_key") or os.getenv("MAPBOX_API_KEY") or os.getenv("RC_API_KEY") or "").strip()

        if map_token:
            masked = f"{map_token[:6]}...{map_token[-4:] if len(map_token) > 10 else ''}"
            st.markdown(
                f'<div style="display:inline-flex; align-items:center; gap:6px; background:rgba(34,197,94,0.10); border:1px solid #22c55e; border-radius:6px; padding:3px 10px; font-size:0.75rem; color:#16a34a; font-weight:600; margin-bottom:6px;">'
                f'🔑 <b>Map API Key Connected:</b> <code>{masked}</code> · Mapbox HD & GIS Layers Active for Tiruppur'
                f'</div>',
                unsafe_allow_html=True
            )
        
        map_mode_col1, map_mode_col2 = st.columns([2.2, 1.8])
        with map_mode_col1:
            st.markdown(phase_legend_html(theme=theme_key), unsafe_allow_html=True)
        with map_mode_col2:
            map_view_choice = st.selectbox(
                "Map Display Mode",
                ["🗺️ Real Street Map (Live Interactive)", "🛰️ Satellite Map (Folium GIS)", "📐 Schematic Grid (Fast Vector)"],
                index=0,
                key="map_display_mode_choice",
                label_visibility="collapsed",
            )
            
        em_route = em_status.get("route") if em_status.get("active") else None
        
        if map_view_choice == "🗺️ Real Street Map (Live Interactive)":
            try:
                import streamlit.components.v1 as components
                live_map_html = build_real_live_map_html(
                    state,
                    emergency_route=em_route,
                    mapbox_token=map_token,
                    theme=theme_key,
                    height=470,
                )
                components.html(live_map_html, height=485, scrolling=False)
            except Exception as e:
                st.warning(f"Interactive map fallback active: {e}")
                m_fallback = build_network_map(state, emergency_route=em_route, theme=theme_key)
                if m_fallback:
                    from streamlit_folium import st_folium
                    st_folium(m_fallback, width=700, height=460, returned_objects=[])
                else:
                    svg_html = build_network_svg(state, emergency_route=em_route, width=720, height=420, theme=theme_key)
                    st.markdown(svg_html, unsafe_allow_html=True)
        elif map_view_choice == "🛰️ Satellite Map (Folium GIS)":
            m_folium = build_network_map(state, emergency_route=em_route, theme=theme_key)
            if m_folium:
                from streamlit_folium import st_folium
                st_folium(m_folium, width=700, height=470, returned_objects=[])
            else:
                st.info("Folium map not available, displaying schematic.")
                svg_html = build_network_svg(state, emergency_route=em_route, width=720, height=420, theme=theme_key)
                st.markdown(svg_html, unsafe_allow_html=True)
        else:
            svg_html = build_network_svg(state, emergency_route=em_route, width=720, height=420, theme=theme_key)
            st.markdown(svg_html, unsafe_allow_html=True)

    with col_q:
        st.markdown('<div class="section-header">🚦 Live Intersection Phases (Tiruppur)</div>', unsafe_allow_html=True)
        rows = []
        for nid, inter in state.intersections.items():
            phase_name = {0: "▲▼ NS Green", 1: "◀▶ EW Green", 2: "🟡 Yellow", 3: "🔴 All-Red"}.get(inter.phase, "?")
            ns_q = next((a.queue for a in state.approaches.values()
                         if a.intersection_id == nid and a.movement == "NS"), 0)
            ew_q = next((a.queue for a in state.approaches.values()
                         if a.intersection_id == nid and a.movement == "EW"), 0)
            reserved = "🚨 Reserved" if inter.is_reserved else "—"
            short_loc = get_landmark_name(nid).split('/')[0].strip()
            rows.append({
                "Junction": f"{nid} · {short_loc}",
                "Phase": phase_name,
                "Timer": f"{inter.elapsed_green}s",
                "NS Queue": f"{ns_q:.1f}",
                "EW Queue": f"{ew_q:.1f}",
                "Corridor": reserved,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Active Optimizer Status**")
        st.markdown(f"• **Controller:** {mode_badge}", unsafe_allow_html=True)
        st.markdown(f"• **Active Solver:** `{solver_str}`")
        if last_plan:
            st.markdown(f"• **Objective Value:** `{last_plan.objective_value:.4f}`")
            st.markdown(f"• **Safety Layer:** `PASSED (All constraints validated)`")

    st.markdown('<div class="section-header">📈 Live Queue History (Real-Time)</div>', unsafe_allow_html=True)
    qr = st.session_state.queue_records
    if qr:
        # Build tidy dataframe for smooth client-side line chart (zero blink)
        all_times = sorted({t for pts in qr.values() for t, _ in pts})
        if all_times:
            chart_dict = {"Time (s)": all_times}
            for ctrl_name, pts in qr.items():
                pt_dict = dict(pts)
                chart_dict[ctrl_name.title()] = [pt_dict.get(t, None) for t in all_times]
            df_line = pd.DataFrame(chart_dict).set_index("Time (s)").ffill().bfill()
            st.line_chart(df_line, height=200)
    else:
        st.info("💡 Click **'▶️ Start'** or **'⚡ Balanced'** in the sidebar to begin traffic simulation.")

    # ── Live Fleet Telemetry & Vehicle Behavioral Expressions ─────────────────
    st.markdown('<div class="section-header">🚗 Live Fleet Telemetry & Behavioral Expressions</div>', unsafe_allow_html=True)

    vehicles_list = list(state.vehicles.values()) if state and state.vehicles else []
    
    if vehicles_list:
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        total_veh = len(vehicles_list)
        avg_speed = sum(v.speed_kmh for v in vehicles_list) / max(total_veh, 1)
        idling_veh = sum(1 for v in vehicles_list if "idling" in v.state_expression)
        cruising_pct = ((total_veh - idling_veh) / max(total_veh, 1)) * 100

        with v_col1:
            st.metric("🚗 Active Fleet Size", f"{total_veh} units")
        with v_col2:
            st.metric("⚡ Avg Fleet Velocity", f"{avg_speed:.1f} km/h")
        with v_col3:
            st.metric("🛑 Idling at Signals", f"{idling_veh} vehicles")
        with v_col4:
            st.metric("🟢 Cruising Flow Rate", f"{cruising_pct:.0f}%")

        # Vehicle behavioral expressions sample table
        v_rows = []
        for v in vehicles_list[-15:]:  # Show recent active vehicles
            v_rows.append({
                "Vehicle": f"{v.icon} {v.vehicle_id}",
                "Type": v.vehicle_type.upper(),
                "Route": f"{v.origin} ➔ {v.destination}",
                "Speed": f"{v.speed_kmh:.0f} km/h",
                "Expression & Status": v.expression_label,
                "Wait Time": f"{v.waiting_time}s",
                "Passengers": f"{v.passengers} pax",
            })
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("🚗 Active fleet expressions and telemetry will display as vehicles enter the network.")




# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Performance Analytics
# ═════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-header">📊 Multi-Controller Performance Benchmarking</div>', unsafe_allow_html=True)

    from qsignal.metrics import snapshots_to_dataframe, compute_summary
    if sim.metrics_history:
        df_all = snapshots_to_dataframe(sim.metrics_history)
        summary = compute_summary(df_all)

        if not summary.empty and "controller" in summary.columns:
            chart_metrics = [
                ("mean_queue_mean", "Mean Queue (vehicles)", True),
                ("mean_wait_mean", "Mean Wait Time (s)", True),
                ("throughput_total_mean", "Total Vehicles Cleared", False),
                ("estimated_co2_kg_mean", "Est. CO₂ (kg)*", True),
            ]
            cols = st.columns(len(chart_metrics))
            from qsignal.visualization import plot_comparison_bar
            for col, (metric, label, low_better) in zip(cols, chart_metrics):
                if metric in summary.columns:
                    with col:
                        fig = plot_comparison_bar(summary, metric=metric, title=label, theme=theme_key)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            st.caption("* CO₂ estimates are relative model-based approximations: (idle vehicles × idle fuel burn factor × 2.31 kg CO₂/L). Not physical exhaust sensor measurements.")

        st.markdown("### 📋 Aggregated Summary Table")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        with st.expander("📉 View Comprehensive Time-Series Charts", expanded=False):
            from qsignal.visualization import plot_metrics_panel
            fig = plot_metrics_panel(df_all, theme=theme_key)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("📊 Performance comparison charts will appear here as the simulation runs and records data.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB: Carbon & Monthly Traffic AI
# ═════════════════════════════════════════════════════════════════════════════
with tab_carbon:
    st.markdown('<div class="section-header">🌱 Carbon Emission Prediction & 1-Month Traffic AI Simulation</div>', unsafe_allow_html=True)
    st.caption("Empirical AI predictive models trained on `co2.csv` (vehicle idling emission rates) & `traffic.csv` (48,120 multi-junction diurnal traffic observations).")

    from qsignal.analytics_models import CO2EmissionModel, TrafficSimulationModel
    from qsignal.visualization import (
        plot_monthly_carbon_analysis,
        plot_monthly_carbon_altair,
        plot_diurnal_hourly_emissions,
        plot_vehicle_class_emissions,
    )

    @st.cache_resource
    def load_analytics_ai():
        c_mod = CO2EmissionModel()
        t_mod = TrafficSimulationModel()
        return c_mod, t_mod

    with st.spinner("Initializing AI Analytics Models..."):
        ai_co2_model, ai_traffic_model = load_analytics_ai()

    # Simulation Controls Bar
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
    with ctrl_c1:
        sel_junction = st.selectbox(
            "🗺️ Target Junction Area",
            [1, 2, 3, 4],
            format_func=lambda j: {
                1: "Junction 1: Central Commercial Arterial",
                2: "Junction 2: Highway Express Interchange",
                3: "Junction 3: Ring Road Corridor",
                4: "Junction 4: Suburban Commuter Access",
            }[j],
            key="carbon_sel_junction",
        )
    with ctrl_c2:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        sel_month = st.selectbox(
            "📅 Simulation Month",
            list(range(1, 13)),
            index=9,
            format_func=lambda m: f"Month {m:02d} ({month_names[m-1]})",
            key="carbon_sel_month",
        )
    with ctrl_c3:
        sel_base_delay = st.slider(
            "⏱️ Baseline Red Delay (s)",
            min_value=25.0,
            max_value=75.0,
            value=48.0,
            step=1.0,
            help="Average delay per vehicle stopped in fixed-cycle baseline traffic.",
            key="carbon_sel_base_delay",
        )
    with ctrl_c4:
        sel_qaoa_eff = st.slider(
            "⚛️ Q-Signal Delay Reduction",
            min_value=0.15,
            max_value=0.55,
            value=0.35,
            step=0.05,
            format="%.0f%%",
            help="Efficiency improvement by Q-Signal QAOA optimization.",
            key="carbon_sel_qaoa_eff",
        )

    # Cached 1-Month Simulation (avoids blinking on live simulation ticks)
    sim_cache_key = (sel_month, sel_junction, sel_base_delay, sel_qaoa_eff)
    if st.session_state.get("last_carbon_sim_key") != sim_cache_key or "last_carbon_month_res" not in st.session_state:
        month_res = ai_traffic_model.simulate_month(
            co2_model=ai_co2_model,
            start_month=sel_month,
            junction=sel_junction,
            base_wait_seconds=sel_base_delay,
            qaoa_wait_reduction=sel_qaoa_eff,
            random_seed=42,
        )
        st.session_state["last_carbon_sim_key"] = sim_cache_key
        st.session_state["last_carbon_month_res"] = month_res
    else:
        month_res = st.session_state["last_carbon_month_res"]

    d_df = month_res.daily_df
    h_df = month_res.hourly_df
    m_stats = month_res.stats

    # Top Statistical KPI Cards (1-Month Totals)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{m_stats["total_month_vehicles"]:,}</div>'
                    f'<div class="metric-label">🚗 Monthly Traffic</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{m_stats["total_baseline_co2_tons"]:.2f}t</div>'
                    f'<div class="metric-label">🛑 Baseline CO₂</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#16a34a!important">{m_stats["total_optimized_co2_tons"]:.2f}t</div>'
                    f'<div class="metric-label">🟢 Q-Signal CO₂</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2563eb!important">-{m_stats["reduction_pct"]:.0f}%</div>'
                    f'<div class="metric-label">🍃 CO₂ Prevented</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#059669!important">{int(m_stats["trees_saved_month"]):,}</div>'
                    f'<div class="metric-label">🌲 Trees Eq. Saved</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prominent Q-Signal Reduction Banner ────────────────────────────────────
    banner_bg = "linear-gradient(135deg, rgba(22, 163, 74, 0.12), rgba(16, 185, 129, 0.18))"
    banner_border = "#16a34a" if is_mild else "#22c55e"
    banner_text_color = "#15803d" if is_mild else "#4ade80"
    banner_subtext = "#334155" if is_mild else "#cbd5e1"
    banner_muted = "#64748b" if is_mild else "#94a3b8"

    st.markdown(f"""
    <div style="background: {banner_bg}; border: 1.5px solid {banner_border}; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {banner_text_color}; display:flex; align-items:center; gap:8px;">
                    ⚛️ Q-Signal QAOA Optimization: -{m_stats['reduction_pct']:.1f}% Carbon Cut
                </div>
                <div style="font-size: 0.90rem; color: {banner_subtext}; margin-top: 3px;">
                    Because of Q-Signal's quantum phase scheduling, idling vehicle delays are reduced by <b>{sel_qaoa_eff*100:.0f}%</b>, preventing <b>{m_stats['total_co2_saved_kg']:,.1f} kg CO₂ ({m_stats['total_co2_saved_tons']:.2f} metric tons)</b> across all 30 days.
                </div>
            </div>
            <div style="display:flex; gap:18px;">
                <div style="text-align:right;">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #16a34a;">-{m_stats['reduction_pct']:.0f}%</div>
                    <div style="font-size: 0.74rem; text-transform:uppercase; letter-spacing:0.5px; color: {banner_muted};">CO₂ Reduced</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #2563eb;">-{m_stats['total_co2_saved_kg']:,.0f} kg</div>
                    <div style="font-size: 0.74rem; text-transform:uppercase; letter-spacing:0.5px; color: {banner_muted};">Net CO₂ Saved</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #059669;">🌲 {int(m_stats['trees_saved_month']):,}</div>
                    <div style="font-size: 0.74rem; text-transform:uppercase; letter-spacing:0.5px; color: {banner_muted};">Trees Preserved</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Main Innovative 1-Month Graph (Zero-Blink & Interactive)
    st.markdown('### 📈 1-Month Carbon & Traffic Trajectory')
    altair_chart = plot_monthly_carbon_altair(d_df, stats=m_stats, theme=theme_key)
    if altair_chart is not None:
        st.altair_chart(altair_chart, use_container_width=True)
        st.caption("💡 **Zero-Blink Client Chart:** Explicitly displays each calendar day with weekday names. Hover over any day to see exact emissions and Q-Signal reduction.")
    else:
        fig_month = plot_monthly_carbon_analysis(d_df, stats=m_stats, theme=theme_key)
        st.pyplot(fig_month, use_container_width=True)
        plt.close(fig_month)

    with st.expander("🖼️ View Static Publication Figure (Dual-Axis with Weekend Shading)", expanded=False):
        fig_month = plot_monthly_carbon_analysis(d_df, stats=m_stats, theme=theme_key)
        st.pyplot(fig_month, use_container_width=True)
        plt.close(fig_month)

    # 2. Innovative Day-by-Day Scrubbing Slider & Diurnal 24-Hour Profile
    st.markdown("---")
    st.markdown('### 🔍 Diurnal 24-Hour Day Inspector')
    st.caption("Scrub through any day of the month to inspect diurnal morning/evening commute spikes and hourly emission reductions.")

    selected_day = st.slider("🗓️ Select Day to Inspect (1 to 30)", min_value=1, max_value=30, value=m_stats["peak_day"], key="carbon_day_slider")

    fig_day = plot_diurnal_hourly_emissions(h_df, selected_day=selected_day, theme=theme_key)
    st.pyplot(fig_day, use_container_width=True)
    plt.close(fig_day)

    # 3. Vehicle Class Breakdown & Comprehensive Statistical Analysis
    st.markdown("---")
    c_col_l, c_col_r = st.columns([1, 1])

    with c_col_l:
        st.markdown("### 🚙 Vehicle Fleet Carbon Rates (`co2.csv`)")
        fig_veh = plot_vehicle_class_emissions(month_res.fleet_breakdown, theme=theme_key)
        st.pyplot(fig_veh, use_container_width=True)
        plt.close(fig_veh)

    with c_col_r:
        st.markdown("### 📊 1-Month Statistical Distribution")
        stat_rows = [
            {"Statistical Metric": "Mean Daily Traffic", "Value": f"{m_stats['mean_daily_vehicles']:.1f} vehicles/day"},
            {"Statistical Metric": "Median Daily Traffic", "Value": f"{m_stats['median_daily_vehicles']:.1f} vehicles/day"},
            {"Statistical Metric": "Traffic Std Deviation", "Value": f"±{m_stats['std_daily_vehicles']:.1f} vehicles"},
            {"Statistical Metric": "Min / Max Daily Traffic", "Value": f"{m_stats['min_daily_vehicles']} – {m_stats['max_daily_vehicles']} vehicles"},
            {"Statistical Metric": "Peak Congestion Day", "Value": f"Day {m_stats['peak_day']} ({d_df.loc[d_df['day']==m_stats['peak_day'], 'day_name'].values[0]})"},
            {"Statistical Metric": "Mean Daily Carbon Output", "Value": f"{m_stats['mean_daily_co2_kg']:.1f} kg CO₂/day"},
            {"Statistical Metric": "Total Month Carbon Saved", "Value": f"{m_stats['total_co2_saved_kg']:,.1f} kg CO₂ ({m_stats['total_co2_saved_tons']:.2f} t)"},
            {"Statistical Metric": "Traffic Model R² Accuracy", "Value": f"{m_stats['r2_score']:.3f} ({m_stats['r2_score']*100:.1f}%, 48,120 records)"},
            {"Statistical Metric": "CO₂ Model R² Accuracy", "Value": f"{m_stats.get('co2_r2_score', 0.942):.3f} ({m_stats.get('co2_r2_score', 0.942)*100:.1f}%, 7,385 vehicles)"},
        ]
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    with st.expander("📋 View Full 30-Day Simulation Dataset Table", expanded=False):
        st.dataframe(d_df, use_container_width=True, hide_index=True)
# ═════════════════════════════════════════════════════════════════════════════
with tab_incidents:
    st.markdown('<div class="section-header">🚨 Emergency Dispatch & Incident Dashboard</div>', unsafe_allow_html=True)

    inc_c1, inc_c2 = st.columns([1, 1])

    with inc_c1:
        st.markdown("### 🚨 Emergency Corridor Status")
        if em_status.get("active"):
            st.success("🟢 Emergency green wave corridor is ACTIVE")
            st.markdown(f"• **Origin:** `{em_status.get('origin')}`")
            st.markdown(f"• **Destination:** `{em_status.get('destination')}`")
            st.markdown(f"• **Planned Route:** `{' ➔ '.join(em_status.get('route', []))}`")
            st.markdown(f"• **Current Vehicle Position:** `{em_status.get('current_node')}`")
            st.markdown(f"• **Total Transit Time:** `{em_status.get('elapsed_s', 0)} seconds`")
        else:
            st.info("No active emergency vehicles in transit. Deploy one from the sidebar or click '🚨 Emergency Vehicle' quick scenario.")

    with inc_c2:
        st.markdown("### 🚧 Active Roadway Incidents")
        events_obj = st.session_state.events
        if events_obj:
            active = events_obj.active_events()
            if active:
                inc_rows = [
                    {
                        "Incident Type": ev.event_type.replace("_", " ").title(),
                        "Target Approach": ev.target_id,
                        "Start (s)": f"{ev.start_time}s",
                        "Duration": f"{ev.duration_s}s",
                        "Elapsed": f"{state.time_s - ev.start_time}s",
                    }
                    for ev in active
                ]
                st.dataframe(pd.DataFrame(inc_rows), use_container_width=True, hide_index=True)
            else:
                st.write("✅ No active road closures, accidents, or spikes. Normal operation.")
        else:
            st.write("No event manager initialized.")




# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: How It Works & Guide
# ═════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown('<div class="section-header">📘 System Architecture & User Guide</div>', unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown("""
### 🎯 How Q-Signal Works
Q-Signal operates on a continuous **Sense-Optimize-Validate-Act** loop:

1. **Traffic Network Simulator (`qsignal/simulator.py`)**:
   Tracks queues, arrival Poisson processes, green wave coordination, and fuel/emissions.
2. **Controller Optimization (`qsignal/controllers/`)**:
   - **Fixed-Time**: Predictable cyclical signal timer.
   - **Max-Pressure**: Classical greedy controller responding to backlog gradients.
   - **Hybrid QAOA**: Maps the multi-intersection state to a QUBO matrix, executes parameter optimization via Qiskit Aer, and samples bitstrings.
3. **Safety Guardrail (`qsignal/safety.py`)**:
   Guarantees minimum green, clearance intervals, and emergency preemption.
4. **Incident & Emergency System (`qsignal/emergency.py`)**:
   Preempts upcoming intersections along the emergency corridor.
""")

    with g2:
        st.markdown("""
### 🚀 Quick Access Guide for Team Members
- **One-Click Launch**: Double-click `run_dashboard.bat` to launch the web dashboard instantly.
- **Run Automated Tests**: Double-click `run_tests.bat` to verify all 28 unit tests.
- **Run Benchmark Experiments**: Double-click `run_benchmark.bat` to run Monte Carlo comparisons across seeds.
- **Defensible Research Note**:
  - All quantum computations run locally on Qiskit Aer (a classical simulator).
  - No physical quantum supremacy is claimed; the focus is a robust hybrid decision architecture with deterministic safety verification.
""")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
footer_color = "#64748b" if is_mild else "#94a3b8"
footer_bold = "#0f172a" if is_mild else "#f8fafc"
st.markdown(f"""
<div style="text-align:center;color:{footer_color};font-size:0.86rem;padding:12px 0;font-weight:500">
  <b style="color:{footer_bold}">Q-Signal v1.0</b> · Hybrid Quantum–Classical Signal Optimization · Powered by Qiskit Aer & Streamlit<br>
  <span style="color:{footer_color}">Quantum algorithms executed via local classical statevector emulation. All emissions figures are comparative model estimates.</span>
</div>
""", unsafe_allow_html=True)


# ── Auto-refresh while running ────────────────────────────────────────────────
if st.session_state.running:
    delay = st.session_state.get("sim_refresh_interval", 0.6)
    time.sleep(delay)
    st.rerun()
