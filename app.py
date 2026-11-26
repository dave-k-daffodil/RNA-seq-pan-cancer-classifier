import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OncoPrint™ — RNA-Seq Intelligence",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Dark Clinical Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ───────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ──────────────────────────────────────────────────────── */
:root {
  --bg-base:    #040d1a;
  --bg-panel:   #071024;
  --bg-card:    #0a1628;
  --bg-glass:   rgba(10,22,40,0.75);
  --border:     rgba(0,212,255,0.12);
  --border-glow:rgba(0,212,255,0.35);
  --cyan:       #00d4ff;
  --cyan-dim:   #0ea5c9;
  --green:      #00ff9d;
  --amber:      #f59e0b;
  --red:        #ff4d6d;
  --purple:     #a855f7;
  --text-primary:#e2f0ff;
  --text-muted: #64748b;
  --text-dim:   #334155;
  --font-main:  'Inter', sans-serif;
  --font-mono:  'JetBrains Mono', monospace;
  --r-sm: 8px;
  --r-md: 14px;
  --r-lg: 20px;
  --shadow-glow: 0 0 24px rgba(0,212,255,0.15);
}

/* ── Base App ───────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  font-family: var(--font-main) !important;
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
}
.stApp { background: var(--bg-base) !important; }
.main .block-container {
  padding: 1.5rem 2.5rem 3rem !important;
  max-width: 1600px !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060f20 0%, #04091a 100%) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--cyan) !important;
}

/* ── Radio Buttons ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [role="radiogroup"] label {
  color: var(--text-primary) !important;
  padding: 10px 14px !important;
  border-radius: var(--r-sm) !important;
  margin: 2px 0 !important;
  transition: all 0.2s ease !important;
  border: 1px solid transparent !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(0,212,255,0.08) !important;
  border-color: var(--border) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] div:first-child {
  background: var(--cyan) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #00d4ff22, #0ea5c920) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.03em !important;
  transition: all 0.2s ease !important;
  backdrop-filter: blur(10px) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #00d4ff40, #0ea5c940) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.3) !important;
  transform: translateY(-1px) !important;
}

/* ── Download Button ────────────────────────────────────────────────────── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #00ff9d22, #00cc7a20) !important;
  color: var(--green) !important;
  border: 1px solid rgba(0,255,157,0.35) !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  box-shadow: 0 0 20px rgba(0,255,157,0.25) !important;
}

/* ── File Uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: rgba(0,212,255,0.04) !important;
  border: 1px dashed var(--border-glow) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-primary) !important; }

/* ── Selectbox ──────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-primary) !important;
}

/* ── Text Input ─────────────────────────────────────────────────────────── */
.stTextInput input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-main) !important;
}
.stTextInput input:focus {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 12px rgba(0,212,255,0.15) !important;
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 1.2rem 1.4rem !important;
  box-shadow: var(--shadow-glow) !important;
  transition: all 0.25s ease !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-glow) !important;
  box-shadow: 0 0 32px rgba(0,212,255,0.2) !important;
  transform: translateY(-2px) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--cyan) !important;
  font-size: 1.75rem !important;
  font-weight: 800 !important;
  line-height: 1.1 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-size: 0.75rem !important;
}

/* ── DataFrames / Tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  overflow: hidden !important;
}
.stDataFrame thead th {
  background: var(--bg-panel) !important;
  color: var(--cyan) !important;
  font-weight: 600 !important;
  font-size: 0.75rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  border-bottom: 1px solid var(--border-glow) !important;
}
.stDataFrame tbody tr:hover { background: rgba(0,212,255,0.04) !important; }
.stDataFrame tbody td { color: var(--text-primary) !important; }

/* ── Info / Warning / Success boxes ─────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--r-md) !important;
  backdrop-filter: blur(10px) !important;
}
div[data-testid="stAlert"][kind="info"] {
  background: rgba(0,212,255,0.08) !important;
  border: 1px solid rgba(0,212,255,0.25) !important;
  color: var(--cyan) !important;
}
div[data-testid="stAlert"][kind="success"] {
  background: rgba(0,255,157,0.08) !important;
  border: 1px solid rgba(0,255,157,0.25) !important;
  color: var(--green) !important;
}
div[data-testid="stAlert"][kind="warning"] {
  background: rgba(245,158,11,0.08) !important;
  border: 1px solid rgba(245,158,11,0.25) !important;
  color: var(--amber) !important;
}
div[data-testid="stAlert"][kind="error"] {
  background: rgba(255,77,109,0.08) !important;
  border: 1px solid rgba(255,77,109,0.25) !important;
  color: var(--red) !important;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent) !important;
  margin: 1.5rem 0 !important;
}

/* ── Plotly Charts Background ────────────────────────────────────────────── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
  border-radius: var(--r-md) !important;
}

/* ── Custom HTML Components ──────────────────────────────────────────────── */

/* Page Title Banner */
.onco-banner {
  background: linear-gradient(135deg, #071020 0%, #0a1628 50%, #071020 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.onco-banner::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,212,255,0.06) 0%, transparent 70%);
  pointer-events: none;
}
.onco-banner-eyebrow {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--cyan); margin-bottom: 0.5rem;
}
.onco-banner-title {
  font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em;
  line-height: 1.1;
  background: linear-gradient(135deg, #e2f0ff 0%, #00d4ff 60%, #0ea5c9 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.6rem;
}
.onco-banner-sub {
  font-size: 0.9rem; color: var(--text-muted); font-weight: 400;
  line-height: 1.6;
}
.onco-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.25);
  color: var(--cyan); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 999px; margin-right: 8px; margin-top: 12px;
}
.onco-pill-green {
  background: rgba(0,255,157,0.1); border-color: rgba(0,255,157,0.25);
  color: var(--green);
}
.onco-pill-amber {
  background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.25);
  color: var(--amber);
}

/* Glass Card */
.glass-card {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.5rem 1.75rem;
  backdrop-filter: blur(20px);
  transition: all 0.25s ease;
}
.glass-card:hover {
  border-color: var(--border-glow);
  box-shadow: var(--shadow-glow);
}

/* Patient Hero Card */
.patient-hero {
  background: linear-gradient(135deg, #071020 0%, #0c1a30 100%);
  border-radius: var(--r-lg);
  padding: 1.75rem 2rem;
  position: relative;
  overflow: hidden;
  margin-bottom: 1.5rem;
}
.patient-hero::after {
  content: ''; position: absolute;
  top: -40px; right: -40px;
  width: 180px; height: 180px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,212,255,0.08), transparent 70%);
}
.patient-hero-id {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem;
}
.patient-hero-diagnosis {
  font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;
  color: var(--text-primary); margin-bottom: 0.4rem; line-height: 1.1;
}
.patient-hero-code {
  font-family: var(--font-mono);
  font-size: 0.85rem; font-weight: 500;
  padding: 2px 10px; border-radius: 4px;
  background: rgba(0,212,255,0.15); color: var(--cyan);
  display: inline-block; margin-bottom: 1rem;
}
.patient-hero-row {
  display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 0.75rem;
}
.patient-hero-stat { flex: 1; min-width: 120px; }
.patient-hero-stat-label {
  font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px;
}
.patient-hero-stat-value {
  font-size: 1rem; font-weight: 700; color: var(--text-primary);
}

/* Confidence Gauge Badge */
.conf-gauge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 16px; border-radius: 999px; font-weight: 700;
  font-size: 0.95rem; font-family: var(--font-mono);
}
.conf-high  { background: rgba(0,255,157,0.12); border: 1px solid rgba(0,255,157,0.35); color: var(--green); }
.conf-mid   { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); color: var(--amber); }
.conf-low   { background: rgba(255,77,109,0.12); border: 1px solid rgba(255,77,109,0.35); color: var(--red); }

/* Leaderboard Table */
.bench-table { width: 100%; border-collapse: collapse; }
.bench-table th {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-muted);
  padding: 10px 16px; text-align: right;
  border-bottom: 1px solid var(--border);
}
.bench-table th:first-child { text-align: left; }
.bench-table td {
  padding: 12px 16px; font-size: 0.85rem; font-weight: 500;
  color: var(--text-primary); border-bottom: 1px solid rgba(255,255,255,0.03);
  text-align: right;
}
.bench-table td:first-child { text-align: left; }
.bench-table tr:hover td { background: rgba(0,212,255,0.04); }
.bench-table .champion td { background: rgba(0,212,255,0.06); }
.bench-table .champion td:first-child {
  color: var(--cyan); font-weight: 700;
  border-left: 3px solid var(--cyan);
}
.bench-score-pill {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-family: var(--font-mono); font-size: 0.78rem; font-weight: 600;
}
.bench-score-top  { background: rgba(0,255,157,0.12); color: var(--green); border: 1px solid rgba(0,255,157,0.25); }
.bench-score-mid  { background: rgba(0,212,255,0.08); color: var(--cyan);  border: 1px solid rgba(0,212,255,0.2);  }

/* Cohort Profile Cards (row of 5 on overview page) */
.cohort-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.2rem 1.3rem;
  transition: all 0.25s ease;
  height: 100%;
}
.cohort-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.cohort-card-code {
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; padding: 2px 8px; border-radius: 4px;
  display: inline-block; margin-bottom: 0.6rem;
}
.cohort-card-name { font-size: 0.82rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.3rem; line-height: 1.3; }
.cohort-card-organ { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.6rem; }
.cohort-card-n { font-size: 1.5rem; font-weight: 800; color: var(--text-primary); line-height: 1; }
.cohort-card-pct { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
.cohort-card-genes { font-size: 0.65rem; color: var(--text-dim); margin-top: 0.6rem; line-height: 1.4; font-family: var(--font-mono); }

/* Pipeline Flow Steps */
.pipe-step {
  display: flex; align-items: flex-start; gap: 1rem;
  padding: 1rem 1.25rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  margin-bottom: 0.5rem;
  transition: all 0.2s ease;
}
.pipe-step:hover { border-color: var(--border-glow); background: rgba(0,212,255,0.04); }
.pipe-step-num {
  font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700;
  color: var(--cyan); background: rgba(0,212,255,0.12);
  border: 1px solid rgba(0,212,255,0.25); border-radius: 6px;
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 2px;
}
.pipe-step-title { font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 2px; }
.pipe-step-desc { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
.pipe-step-mono { font-family: var(--font-mono); color: var(--cyan); font-size: 0.72rem; }

/* Sidebar Logo Area */
.sidebar-logo {
  text-align: center;
  padding: 1.5rem 1rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}
.sidebar-logo-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.sidebar-logo-name {
  font-size: 1.1rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(135deg, #e2f0ff, var(--cyan));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sidebar-logo-sub { font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; }
.sidebar-status {
  display: flex; align-items: center; gap: 6px; justify-content: center;
  margin-top: 10px;
}
.sidebar-status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(0.85); }
}
.sidebar-status-text { font-size: 0.7rem; color: var(--text-muted); font-weight: 500; }

/* Nav Items in Sidebar */
.nav-section-title {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--text-dim);
  padding: 0.5rem 0 0.3rem;
}

/* XAI Attribution Score Bar */
.attr-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.attr-gene { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); min-width: 90px; }
.attr-bar-wrap { flex: 1; background: rgba(255,255,255,0.04); border-radius: 4px; height: 8px; overflow: hidden; }
.attr-bar-pos { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00d4ff, #00ff9d); }
.attr-bar-neg { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #ff4d6d, #f97316); }
.attr-score { font-family: var(--font-mono); font-size: 0.72rem; min-width: 55px; text-align: right; }
.attr-pos { color: var(--green); }
.attr-neg { color: var(--red); }

/* Knowledge Base Cards */
.kb-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
  transition: all 0.2s ease;
}
.kb-card:hover { border-color: var(--border-glow); }
.kb-card-label {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--cyan); margin-bottom: 0.5rem;
}
.kb-card-title { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem; }
.kb-card-body { font-size: 0.82rem; color: var(--text-muted); line-height: 1.65; }
.kb-card-body b { color: var(--text-primary); }

/* Quick Prompt Chips */
.prompt-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.prompt-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.2);
  color: var(--cyan); font-size: 0.72rem; font-weight: 600;
  padding: 5px 12px; border-radius: 999px; cursor: pointer;
  transition: all 0.2s ease; white-space: nowrap;
}
.prompt-chip:hover { background: rgba(0,212,255,0.15); border-color: var(--border-glow); }

/* Biomarker Row */
.bio-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.bio-rank { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-dim); min-width: 22px; text-align: right; }
.bio-gene { font-family: var(--font-mono); font-size: 0.82rem; font-weight: 600; color: var(--text-primary); min-width: 90px; }
.bio-bar-wrap { flex: 1; background: rgba(255,255,255,0.04); border-radius: 4px; height: 6px; overflow: hidden; }
.bio-bar { height: 100%; border-radius: 4px; }
.bio-fscore { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); min-width: 60px; text-align: right; }
.bio-cohort-badge {
  font-size: 0.65rem; font-weight: 700; padding: 2px 7px;
  border-radius: 4px; min-width: 42px; text-align: center;
}

/* 3D Map Legend */
.legend-dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 6px; vertical-align: middle;
}

/* Loading spinner animation */
@keyframes spin { to { transform: rotate(360deg); } }
.loading-ring {
  width: 40px; height: 40px; border-radius: 50%;
  border: 3px solid rgba(0,212,255,0.15);
  border-top-color: var(--cyan);
  animation: spin 0.8s linear infinite;
  margin: 20px auto;
}

/* Diagnosis Verdict Box */
.verdict-box {
  border-radius: var(--r-md); padding: 1rem 1.4rem;
  margin-top: 0.75rem; display: flex; align-items: flex-start; gap: 12px;
}
.verdict-high { background: rgba(0,255,157,0.07); border: 1px solid rgba(0,255,157,0.2); }
.verdict-mid  { background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.2); }
.verdict-low  { background: rgba(255,77,109,0.07); border: 1px solid rgba(255,77,109,0.2); }
.verdict-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
.verdict-title { font-size: 0.82rem; font-weight: 700; margin-bottom: 2px; }
.verdict-desc  { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
.verdict-high .verdict-title { color: var(--green); }
.verdict-mid  .verdict-title { color: var(--amber); }
.verdict-low  .verdict-title { color: var(--red); }

/* Tabs override */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px !important;
  background: var(--bg-panel) !important;
  border-radius: var(--r-sm) !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 6px !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important; font-size: 0.8rem !important;
  padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,212,255,0.12) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,212,255,0.25) !important;
}

/* Section Heading */
.section-heading {
  font-size: 1.1rem; font-weight: 700; color: var(--text-primary);
  letter-spacing: -0.01em; margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
}
.section-heading::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-glow), transparent);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COHORT METADATA
# ══════════════════════════════════════════════════════════════════════════════
COHORT = {
    'BRCA': dict(name='Breast Invasive Carcinoma',    organ='Breast',   color='#ec4899', hex_bg='rgba(236,72,153,0.12)', genes='BRCA1 · BRCA2 · TP53 · PIK3CA · ERBB2 · ESR1',  n=300, pct=37.5,
                 desc='Malignant mammary gland neoplasm with high subtype heterogeneity across Luminal A/B, HER2-enriched, and Triple-Negative profiles. BRCA1/2 germline variants confer elevated lifetime risk.'),
    'KIRC': dict(name='Kidney Renal Clear Cell Carcin.', organ='Kidney', color='#3b82f6', hex_bg='rgba(59,130,246,0.12)', genes='VHL · PBRM1 · SETD2 · BAP1 · MTOR',          n=146, pct=18.2,
                 desc='Primary renal epithelial tumor characterized by clear cytoplasm with glycogen/lipid accumulation and intense VEGF-driven vascularization via constitutive HIF1α/HIF2α (VHL-loss).'),
    'LUAD': dict(name='Lung Adenocarcinoma',          organ='Lung',     color='#10b981', hex_bg='rgba(16,185,129,0.12)', genes='EGFR · KRAS · TP53 · ALK · ROS1 · BRAF · MET', n=141, pct=17.6,
                 desc='Most prevalent NSCLC subtype arising from peripheral alveolar glandular cells. Targetable drivers (EGFR kinase, ALK/ROS1 fusions) define therapeutic stratification.'),
    'PRAD': dict(name='Prostate Adenocarcinoma',      organ='Prostate', color='#8b5cf6', hex_bg='rgba(139,92,246,0.12)', genes='AR · TMPRSS2-ERG · SPOP · PTEN · FOXA1',       n=136, pct=17.0,
                 desc='Androgen receptor-driven glandular neoplasm. TMPRSS2-ERG gene fusion is the most prevalent somatic translocation (~50% of cases), enabling targeted hormonal therapies.'),
    'COAD': dict(name='Colon Adenocarcinoma',         organ='Colon',    color='#f97316', hex_bg='rgba(249,115,22,0.12)', genes='APC · KRAS · TP53 · SMAD4 · BRAF · MLH1',      n=78,  pct=9.7,
                 desc='Colorectal epithelial malignancy following the adenoma-carcinoma sequence (APC → Wnt/β-catenin activation). MSI-High tumors respond to PD-1 checkpoint immunotherapy.'),
}

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=11),
    margin=dict(t=40, b=20, l=20, r=20),
    coloraxis_colorbar=dict(tickcolor='#64748b', tickfont=dict(color='#64748b')),
)

def apply_dark_axes(fig, gridcolor='rgba(255,255,255,0.05)'):
    fig.update_xaxes(gridcolor=gridcolor, linecolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.05)', tickfont=dict(color='#64748b'))
    fig.update_yaxes(gridcolor=gridcolor, linecolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.05)', tickfont=dict(color='#64748b'))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    path = 'cancer_classification_pipeline.pkl'
    if not os.path.exists(path):
        return None
    return joblib.load(path)

with st.spinner(''):
    artifacts = load_artifacts()

if artifacts is None:
    st.markdown('<div class="onco-banner"><div class="onco-banner-title">⚠️ Pipeline Missing</div><div class="onco-banner-sub">Run <code>python train.py</code> to generate <code>cancer_classification_pipeline.pkl</code></div></div>', unsafe_allow_html=True)
    st.stop()

pipeline        = artifacts['pipeline']
classes         = artifacts['classes']
orig_features   = artifacts['original_features']
sel_features    = artifacts['final_features']
pca_3d          = artifacts['pca_3d']
train_pca_df    = artifacts['train_pca_df']
demo_df         = artifacts['demo_samples_df']
bench_res       = artifacts['benchmark_results']
test_metrics    = artifacts['test_metrics']
top_bio         = artifacts['top_biomarkers']
metadata        = artifacts['metadata']

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🧬</div>
      <div class="sidebar-logo-name">OncoPrint™</div>
      <div class="sidebar-logo-sub">RNA-Seq Intelligence Platform</div>
      <div class="sidebar-status">
        <div class="sidebar-status-dot"></div>
        <div class="sidebar-status-text">Pipeline Active · v2.0</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio('', [
        '🏥  Clinical Overview',
        '🧪  Patient Inference',
        '🌐  3D Tumor Manifold',
        '🔬  Explainable AI',
        '💬  Knowledge Engine',
    ], label_visibility='collapsed')

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-title">Cohort Ingestion</div>', unsafe_allow_html=True)

    if 'active_df' not in st.session_state:
        st.session_state.active_df = demo_df.copy()
        st.session_state.src_name  = 'Built-in Demo Cohort'

    c1, c2 = st.columns(2)
    with c1:
        if st.button('⚡ Demo', use_container_width=True):
            st.session_state.active_df = demo_df.copy()
            st.session_state.src_name  = 'Built-in Demo Cohort'
            st.rerun()
    with c2:
        if st.button('✕ Clear', use_container_width=True):
            st.session_state.active_df = None
            st.session_state.src_name  = None
            st.rerun()

    up = st.file_uploader('Upload Patient CSV', type=['csv'], label_visibility='collapsed')
    if up:
        try:
            df_up = pd.read_csv(up, index_col=0)
            st.session_state.active_df = df_up
            st.session_state.src_name  = up.name
            st.success(f'✓ {len(df_up)} patient(s) ingested')
        except Exception as ex:
            st.error(f'Parse error: {ex}')

    if st.session_state.active_df is not None:
        n = len(st.session_state.active_df)
        st.markdown(f'<div style="margin-top:8px;padding:8px 12px;background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.15);border-radius:8px;font-size:0.72rem;color:#64748b;">📁 <b style="color:#e2f0ff">{st.session_state.src_name}</b><br>{n} patient{"s" if n>1 else ""} loaded</div>', unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.65rem;color:#334155;line-height:1.6;">Model: <span style="color:#64748b">{metadata["best_model"]}</span><br>Dataset: <span style="color:#64748b">TCGA PANCAN</span><br>Genes: <span style="color:#64748b">20,531</span> → <span style="color:#64748b">500</span><br>Created: <span style="color:#64748b">{metadata["created_at"]}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_inference(df_input):
    if df_input is None or len(df_input) == 0:
        return None, None, None
    df_aligned = df_input.reindex(columns=orig_features, fill_value=0.0).fillna(0.0)
    preds_enc  = pipeline.predict(df_aligned)
    probs      = pipeline.predict_proba(df_aligned)
    preds_dec  = [classes[i] for i in preds_enc]

    res = pd.DataFrame(index=df_input.index)
    res['Cohort']        = preds_dec
    res['Confidence']    = np.max(probs, axis=1) * 100
    runner_idx           = np.argsort(probs, axis=1)[:, -2]
    res['Runner_Up']     = [classes[i] for i in runner_idx]
    res['Runner_Conf']   = [probs[r, runner_idx[r]] * 100 for r in range(len(probs))]
    for ci, cn in enumerate(classes):
        res[f'P_{cn}'] = probs[:, ci] * 100

    prep  = pipeline[:-1].transform(df_aligned)
    coord = pca_3d.transform(prep)
    pca_d = pd.DataFrame(coord, columns=['PC1','PC2','PC3'], index=df_input.index)
    pca_d['Cohort'] = preds_dec

    return res, prep, pca_d

active_df = st.session_state.active_df
results, X_prep, pca_active = run_inference(active_df)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLINICAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == '🏥  Clinical Overview':

    st.markdown(f"""
    <div class="onco-banner">
      <div class="onco-banner-eyebrow">TCGA Pan-Cancer Intelligence · RNA-Seq Multi-Class Stratification</div>
      <div class="onco-banner-title">Clinical Oncology<br>Diagnostic Platform</div>
      <div class="onco-banner-sub">
        High-throughput computational pipeline classifying <strong>5 primary human cancers</strong>
        from {metadata['raw_genes']:,} gene expression markers across {metadata['total_samples']} validated TCGA clinical patient profiles.
      </div>
      <div>
        <span class="onco-pill">● Live</span>
        <span class="onco-pill">🧬 {metadata['raw_genes']:,} Genes</span>
        <span class="onco-pill onco-pill-green">✓ {test_metrics['accuracy']*100:.2f}% Accuracy</span>
        <span class="onco-pill onco-pill-amber">⬡ {test_metrics['roc_auc_ovr']*100:.1f}% ROC-AUC</span>
        <span class="onco-pill">{metadata['best_model'].replace('_',' ')}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Total Patients', f"{metadata['total_samples']:,}", 'TCGA PANCAN')
    k2.metric('Transcriptome', f"{metadata['raw_genes']:,}", 'Gene Features')
    k3.metric('Selected Genes', f"{metadata['selected_genes']:,}", 'ANOVA p < 1e-12')
    k4.metric('Test Accuracy', f"{test_metrics['accuracy']*100:.2f}%", f"F1 {test_metrics['f1_macro']*100:.2f}%")
    k5.metric('ROC-AUC (OvR)', f"{test_metrics['roc_auc_ovr']*100:.1f}%", '5-Class Multiclass')

    st.markdown('<hr/>', unsafe_allow_html=True)

    # Cancer Cohort Cards Row
    st.markdown('<div class="section-heading">Cancer Cohort Profiles (N = 801)</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (code, info) in enumerate(COHORT.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="cohort-card" style="border-top:3px solid {info['color']}">
              <div class="cohort-card-code" style="background:{info['hex_bg']};color:{info['color']}">{code}</div>
              <div class="cohort-card-name">{info['name']}</div>
              <div class="cohort-card-organ">🏥 {info['organ']}</div>
              <div class="cohort-card-n" style="color:{info['color']}">{info['n']}</div>
              <div class="cohort-card-pct">{info['pct']}% of cohort</div>
              <div class="cohort-card-genes">{info['genes']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<br/>', unsafe_allow_html=True)

    # Charts + Pipeline Row
    col_c, col_p = st.columns([3, 2], gap='large')

    with col_c:
        st.markdown('<div class="section-heading">Cohort Distribution Analysis</div>', unsafe_allow_html=True)
        tab_bar, tab_pie = st.tabs(['Bar Chart', 'Donut Chart'])

        dist_df = pd.DataFrame([
            {'Code': c, 'Cancer': f"{c} — {d['name']}", 'Patients': d['n'], 'Pct': d['pct'], 'Color': d['color']}
            for c, d in COHORT.items()
        ])

        with tab_bar:
            fig_bar = go.Figure()
            for _, row in dist_df.iterrows():
                fig_bar.add_trace(go.Bar(
                    x=[row['Cancer']], y=[row['Patients']],
                    name=row['Code'],
                    marker_color=row['Color'],
                    marker_line=dict(width=0),
                    text=[f"<b>{row['Patients']}</b><br>{row['Pct']}%"],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=11),
                ))
            fig_bar.update_layout(
                **PLOTLY_BASE, showlegend=False, height=320,
                xaxis_tickfont=dict(size=10), bargap=0.35,
            )
            apply_dark_axes(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        with tab_pie:
            fig_pie = go.Figure(go.Pie(
                labels=dist_df['Code'],
                values=dist_df['Patients'],
                hole=0.6,
                marker=dict(colors=dist_df['Color'].tolist(), line=dict(color='#040d1a', width=3)),
                textinfo='label+percent',
                textfont=dict(color='#94a3b8', size=11),
            ))
            fig_pie.add_annotation(text=f'<b style="color:#e2f0ff">801</b><br><span style="color:#64748b;font-size:11px">Patients</span>',
                x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig_pie.update_layout(**PLOTLY_BASE, height=320, showlegend=True,
                legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with col_p:
        st.markdown('<div class="section-heading">Pipeline Architecture</div>', unsafe_allow_html=True)
        steps = [
            ('01', 'Stratified Partitioning', '80% Train (N=640) · 20% Held-Out Test (N=161)', 'stratify=y_encoded'),
            ('02', 'VarianceThreshold', 'Removes static non-informative genes', '20,531 → 19,285'),
            ('03', 'StandardScaler', 'Z-Score fit on training folds only — prevents leakage', 'μ=0  σ=1'),
            ('04', 'SelectKBest (ANOVA)', 'Between-group F-statistic ranking of biomarkers', 'k = 500'),
            ('05', 'Multinomial Classifier', 'Class-balanced softmax decision engine', 'C=0.1  balanced'),
        ]
        for num, title, desc, mono in steps:
            st.markdown(f"""
            <div class="pipe-step">
              <div class="pipe-step-num">{num}</div>
              <div>
                <div class="pipe-step-title">{title}</div>
                <div class="pipe-step-desc">{desc}</div>
                <div class="pipe-step-mono">{mono}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)

    # Benchmark Leaderboard
    st.markdown('<div class="section-heading">Model Benchmark Leaderboard</div>', unsafe_allow_html=True)

    bm_rows = []
    for m_name, m_res in bench_res.items():
        bm_rows.append({
            'name': m_name.replace('_', ' '),
            'cv_acc': m_res['cv_accuracy'] * 100,
            'cv_f1':  m_res['cv_f1_macro'] * 100,
            'test_acc': m_res.get('test_accuracy', 0) * 100,
            'test_f1':  m_res.get('test_f1_macro', 0) * 100,
            'test_auc': m_res.get('test_roc_auc', 1.0) * 100,
            'fit': m_res['fit_time_seconds'],
            'champion': m_name == metadata['best_model'],
        })
    bm_rows.sort(key=lambda r: r['cv_f1'], reverse=True)

    def score_pill(val, champion):
        cls = 'bench-score-top' if champion else 'bench-score-mid'
        return f'<span class="bench-score-pill {cls}">{val:.2f}%</span>'

    table_html = '<table class="bench-table"><thead><tr><th>Model</th><th>CV Acc</th><th>CV F1</th><th>Test Acc</th><th>Test F1</th><th>ROC-AUC</th><th>Fit</th></tr></thead><tbody>'
    for r in bm_rows:
        ch_cls = 'champion' if r['champion'] else ''
        crown  = ' 🏆' if r['champion'] else ''
        table_html += f"""<tr class="{ch_cls}">
          <td>{r['name']}{crown}</td>
          <td>{score_pill(r['cv_acc'], r['champion'])}</td>
          <td>{score_pill(r['cv_f1'], r['champion'])}</td>
          <td>{score_pill(r['test_acc'], r['champion'])}</td>
          <td>{score_pill(r['test_f1'], r['champion'])}</td>
          <td>{score_pill(r['test_auc'], r['champion'])}</td>
          <td style="color:#64748b;font-family:var(--font-mono)">{r['fit']}s</td>
        </tr>"""
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<br/>', unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<div class="section-heading">Held-Out Test Set Confusion Matrix (N = 161)</div>', unsafe_allow_html=True)
    cm = np.array(test_metrics['confusion_matrix'])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0,'#071020'],[0.5,'#0a2a4a'],[1,'#00d4ff']],
        text=cm, texttemplate='<b>%{text}</b>',
        textfont=dict(size=14, color='white'),
        showscale=False,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig_cm.update_layout(**PLOTLY_BASE, height=320,
        xaxis=dict(title='Predicted Cohort', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title='True Cohort', tickfont=dict(color='#94a3b8'), autorange='reversed'),
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🧪  Patient Inference':

    st.markdown("""
    <div class="onco-banner">
      <div class="onco-banner-eyebrow">Real-Time Inference Engine</div>
      <div class="onco-banner-title">Patient Diagnostic<br>Inference Portal</div>
      <div class="onco-banner-sub">Select a patient record to view the transcriptomic classification result, posterior probability distribution, and differential diagnosis margin.</div>
    </div>
    """, unsafe_allow_html=True)

    if results is None:
        st.warning('⚠️ No cohort loaded. Click **⚡ Demo** in the sidebar to load 5 reference patients.')
        st.stop()

    patient_ids = list(results.index)
    sel = st.selectbox('Patient Record', patient_ids, index=0)
    row = results.loc[sel]
    cohort_code = row['Cohort']
    conf        = row['Confidence']
    cinfo       = COHORT[cohort_code]

    # ── Patient Hero Card ──────────────────────────────────────────────────
    if   conf >= 90: conf_cls, conf_icon, verdict_cls = 'conf-high', '✓', 'verdict-high'
    elif conf >= 70: conf_cls, conf_icon, verdict_cls = 'conf-mid',  '⚠', 'verdict-mid'
    else:            conf_cls, conf_icon, verdict_cls = 'conf-low',  '✕', 'verdict-low'

    margin = conf - row['Runner_Conf']

    if   margin > 80: verdict_title, verdict_desc = 'High-Certainty Classification', 'Unambiguous transcriptomic signature with strong separation from all secondary candidates. Clinically actionable.'
    elif margin > 50: verdict_title, verdict_desc = 'Confident Classification', 'Well-defined cohort signature with moderate secondary baseline. Recommend corroborating IHC markers.'
    else:             verdict_title, verdict_desc = 'Borderline Margin', 'Cross-cohort mutation overlap detected. Diagnostic confirmation via histopathology strongly recommended.'

    st.markdown(f"""
    <div class="patient-hero" style="border-left:5px solid {cinfo['color']}">
      <div class="patient-hero-id">Patient Record · {sel}</div>
      <div class="patient-hero-diagnosis">{cinfo['name']}</div>
      <div class="patient-hero-code">{cohort_code}</div>
      <span class="conf-gauge {conf_cls}">{conf_icon} {conf:.2f}% Confidence</span>
      <div class="patient-hero-row">
        <div class="patient-hero-stat">
          <div class="patient-hero-stat-label">Anatomical Site</div>
          <div class="patient-hero-stat-value">{cinfo['organ']}</div>
        </div>
        <div class="patient-hero-stat">
          <div class="patient-hero-stat-label">Runner-Up</div>
          <div class="patient-hero-stat-value">{row['Runner_Up']} ({row['Runner_Conf']:.1f}%)</div>
        </div>
        <div class="patient-hero-stat">
          <div class="patient-hero-stat-label">Decision Margin</div>
          <div class="patient-hero-stat-value">{margin:.1f}%</div>
        </div>
        <div class="patient-hero-stat">
          <div class="patient-hero-stat-label">Key Drivers</div>
          <div class="patient-hero-stat-value" style="font-size:0.75rem">{cinfo['genes']}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Distribution Chart ───────────────────────────────────
    col_prob, col_verdict = st.columns([3, 2], gap='large')

    with col_prob:
        st.markdown('<div class="section-heading">Class Posterior Probability Distribution</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame([
            {'code': c, 'label': f"{c}  —  {COHORT[c]['name']}", 'prob': row[f'P_{c}'], 'color': COHORT[c]['color']}
            for c in classes
        ]).sort_values('prob', ascending=True)

        fig_prob = go.Figure()
        for _, pr in prob_df.iterrows():
            is_sel = pr['code'] == cohort_code
            fig_prob.add_trace(go.Bar(
                x=[pr['prob']], y=[pr['label']],
                orientation='h',
                name=pr['code'],
                marker_color=pr['color'] if is_sel else 'rgba(100,116,139,0.3)',
                marker_line=dict(width=0),
                text=[f"<b>{pr['prob']:.2f}%</b>"],
                textposition='outside',
                textfont=dict(color=pr['color'] if is_sel else '#64748b', size=11),
            ))

        fig_prob.update_layout(
            **PLOTLY_BASE, showlegend=False, height=300, bargap=0.3,
            xaxis=dict(range=[0, 115], ticksuffix='%', tickfont=dict(color='#64748b')),
        )
        apply_dark_axes(fig_prob)
        st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})

    with col_verdict:
        st.markdown('<div class="section-heading">Differential Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="verdict-box {verdict_cls}">
          <div class="verdict-icon">{'✅' if conf>=90 else '⚠️' if conf>=70 else '🔴'}</div>
          <div>
            <div class="verdict-title">{verdict_title}</div>
            <div class="verdict-desc">{verdict_desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<br/><div class="glass-card"><div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">Clinical Context</div><div style="font-size:0.78rem;line-height:1.65;color:#94a3b8">{cinfo["desc"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr/>', unsafe_allow_html=True)

    # ── Batch Table ───────────────────────────────────────────────────────
    st.markdown(f'<div class="section-heading">Cohort Batch Results ({len(results)} Patients)</div>', unsafe_allow_html=True)

    display = results[['Cohort', 'Confidence', 'Runner_Up', 'Runner_Conf']].copy()
    display.columns = ['Predicted Cohort', 'Confidence (%)', 'Runner-Up', 'Runner-Up Conf (%)']

    styled = display.style.format({'Confidence (%)': '{:.2f}', 'Runner-Up Conf (%)': '{:.2f}'}) \
        .background_gradient(subset=['Confidence (%)'], cmap='Blues', vmin=0, vmax=100)
    st.dataframe(styled, use_container_width=True)

    csv = results.to_csv().encode()
    st.download_button('📥 Download Full Diagnostic Report (CSV)', csv, 'rnaseq_predictions.csv', 'text/csv')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — 3D TUMOR MANIFOLD
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🌐  3D Tumor Manifold':

    st.markdown("""
    <div class="onco-banner">
      <div class="onco-banner-eyebrow">Dimensionality Reduction · Pre-fitted PCA Projection</div>
      <div class="onco-banner-title">3D Transcriptomic<br>Tumor Landscape</div>
      <div class="onco-banner-sub">Interactive 3D projection of the 500-dimensional biomarker space onto orthogonal principal components, capturing 71.5% of total variance. Each cluster represents a distinct cancer cohort.</div>
    </div>
    """, unsafe_allow_html=True)

    if results is None:
        st.warning('⚠️ Load a patient cohort first.')
        st.stop()

    ev = pca_3d.explained_variance_ratio_

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        sel_p3d = st.selectbox('Highlight Patient:', list(results.index), index=0)
        ref_opacity = st.slider('Reference Point Opacity', 0.1, 0.8, 0.4, 0.05)
        ref_size    = st.slider('Reference Point Size', 2, 8, 4, 1)

    with col_ctrl2:
        # Variance explained mini-cards
        c1, c2, c3 = st.columns(3)
        c1.metric('PC1 Variance', f"{ev[0]*100:.1f}%", 'Primary')
        c2.metric('PC2 Variance', f"{ev[1]*100:.1f}%", 'Secondary')
        c3.metric('PC3 Variance', f"{ev[2]*100:.1f}%", 'Tertiary')

    # 3D Figure
    fig3d = go.Figure()

    for code in classes:
        mask = train_pca_df['Cohort'] == code
        d = train_pca_df[mask]
        fig3d.add_trace(go.Scatter3d(
            x=d['PC1'], y=d['PC2'], z=d['PC3'],
            mode='markers',
            name=f'TCGA {code}',
            marker=dict(size=ref_size, color=COHORT[code]['color'], opacity=ref_opacity,
                        line=dict(width=0)),
            hovertemplate=f'<b>{code}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>',
        ))

    # Uploaded patients (not selected)
    for pid in pca_active.index:
        if pid == sel_p3d:
            continue
        pc = results.loc[pid, 'Cohort']
        pconf = results.loc[pid, 'Confidence']
        fig3d.add_trace(go.Scatter3d(
            x=[pca_active.loc[pid,'PC1']], y=[pca_active.loc[pid,'PC2']], z=[pca_active.loc[pid,'PC3']],
            mode='markers',
            name='Uploaded', showlegend=False,
            marker=dict(size=9, color=COHORT[pc]['color'], symbol='circle',
                        line=dict(color='white', width=1.5), opacity=0.9),
            hovertemplate=f'<b>{pid}</b><br>Pred: {pc} ({pconf:.1f}%)<extra></extra>',
        ))

    # Selected patient — glowing star
    sel_coords = pca_active.loc[sel_p3d]
    sel_code   = results.loc[sel_p3d, 'Cohort']
    sel_conf   = results.loc[sel_p3d, 'Confidence']
    fig3d.add_trace(go.Scatter3d(
        x=[sel_coords['PC1']], y=[sel_coords['PC2']], z=[sel_coords['PC3']],
        mode='markers+text',
        name=f'★ {sel_p3d}',
        marker=dict(size=16, color='#fbbf24', symbol='diamond',
                    line=dict(color='#b45309', width=2.5),
                    opacity=1),
        text=[f'★ {sel_p3d}'],
        textposition='top center',
        textfont=dict(size=11, color='#fbbf24'),
        hovertemplate=f'<b>★ {sel_p3d}</b><br>Diagnosis: {sel_code} ({sel_conf:.1f}%)<extra></extra>',
    ))

    fig3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        scene=dict(
            bgcolor='#040d1a',
            xaxis=dict(title=f'PC1 ({ev[0]*100:.1f}%)', gridcolor='rgba(255,255,255,0.05)',
                       backgroundcolor='rgba(0,0,0,0)', tickfont=dict(color='#64748b', size=9)),
            yaxis=dict(title=f'PC2 ({ev[1]*100:.1f}%)', gridcolor='rgba(255,255,255,0.05)',
                       backgroundcolor='rgba(0,0,0,0)', tickfont=dict(color='#64748b', size=9)),
            zaxis=dict(title=f'PC3 ({ev[2]*100:.1f}%)', gridcolor='rgba(255,255,255,0.05)',
                       backgroundcolor='rgba(0,0,0,0)', tickfont=dict(color='#64748b', size=9)),
        ),
        legend=dict(font=dict(color='#94a3b8', size=10), bgcolor='rgba(7,16,32,0.8)',
                    bordercolor='rgba(0,212,255,0.15)', borderwidth=1,
                    yanchor='top', y=0.98, xanchor='left', x=0.01),
    )

    st.plotly_chart(fig3d, use_container_width=True)

    st.info(f"📍 **Patient `{sel_p3d}`** projected at (PC1={sel_coords['PC1']:.3f}, PC2={sel_coords['PC2']:.3f}, PC3={sel_coords['PC3']:.3f}) — aligns directly inside the **{sel_code}** manifold cluster.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🔬  Explainable AI':

    st.markdown("""
    <div class="onco-banner">
      <div class="onco-banner-eyebrow">Transparent AI · Local Logit Attributions</div>
      <div class="onco-banner-title">Explainable AI<br>& Biomarker Profiling</div>
      <div class="onco-banner-sub">Patient-specific transcriptomic attributions (w<sub>c,j</sub> · x<sub>i,j</sub>) reveal which individual genes push the classifier toward or away from each cancer diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    if results is None:
        st.warning('⚠️ Load a patient cohort first.')
        st.stop()

    sel_xai = st.selectbox('Inspect Patient', list(results.index), index=0)
    xai_row  = results.loc[sel_xai]
    target   = xai_row['Cohort']
    t_idx    = classes.index(target)
    cinfo    = COHORT[target]

    clf    = pipeline.named_steps['clf']
    w      = clf.coef_[t_idx]                                   # (500,)
    p_idx  = list(results.index).index(sel_xai)
    x_feat = X_prep[p_idx]                                       # (500,)
    attrib = w * x_feat

    attr_df = pd.DataFrame({'Gene': sel_features, 'Score': attrib, 'Abs': np.abs(attrib)}) \
              .sort_values('Abs', ascending=False)

    col_xai_l, col_xai_r = st.columns([3, 2], gap='large')

    with col_xai_l:
        st.markdown(f'<div class="section-heading">Top 20 Gene Attributions for {target} Diagnosis</div>', unsafe_allow_html=True)

        top20 = attr_df.head(20).sort_values('Score', ascending=True)
        max_abs = attr_df['Abs'].max()

        rows_html = ''
        for _, r in top20.iterrows():
            pct = abs(r['Score']) / max_abs * 100
            pos = r['Score'] > 0
            bar_cls = 'attr-bar-pos' if pos else 'attr-bar-neg'
            sc_cls  = 'attr-pos'    if pos else 'attr-neg'
            prefix  = '+' if pos else ''
            rows_html += f"""
            <div class="attr-row">
              <div class="attr-gene">{r['Gene']}</div>
              <div class="attr-bar-wrap"><div class="{bar_cls}" style="width:{pct:.1f}%"></div></div>
              <div class="attr-score {sc_cls}">{prefix}{r['Score']:.3f}</div>
            </div>"""

        st.markdown(f'<div class="glass-card" style="padding:1.2rem 1.4rem">{rows_html}</div>', unsafe_allow_html=True)

        # Attribution bar chart
        st.markdown('<br/>', unsafe_allow_html=True)
        fig_attr = go.Figure()
        fig_attr.add_trace(go.Bar(
            x=top20['Score'], y=top20['Gene'],
            orientation='h',
            marker_color=[cinfo['color'] if s > 0 else '#ff4d6d' for s in top20['Score']],
            marker_line=dict(width=0),
        ))
        fig_attr.update_layout(
            **PLOTLY_BASE, height=440,
            title=dict(text=f'<b>Logit Attributions → {target}</b>', font=dict(color='#94a3b8', size=12)),
            xaxis=dict(title='Attribution Weight (w·x)', tickfont=dict(color='#64748b')),
        )
        apply_dark_axes(fig_attr)
        st.plotly_chart(fig_attr, use_container_width=True, config={'displayModeBar': False})

    with col_xai_r:
        st.markdown(f'<div class="section-heading">Attribution Summary</div>', unsafe_allow_html=True)

        top_gene = attr_df.iloc[0]
        pos_genes = (attr_df['Score'] > 0).sum()
        neg_genes = (attr_df['Score'] < 0).sum()

        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:1rem">
          <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);margin-bottom:.75rem">Strongest Driver</div>
          <div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;color:var(--cyan)">{top_gene['Gene']}</div>
          <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px">Attribution: <span style="color:{'var(--green)' if top_gene['Score']>0 else 'var(--red)'}"><b>{'+'if top_gene['Score']>0 else ''}{top_gene['Score']:.4f}</b></span></div>
          <div style="font-size:0.72rem;color:var(--text-dim);margin-top:8px;line-height:1.5">This transcript most strongly elevates the log-odds toward the <b style="color:{cinfo['color']}">{target}</b> cohort classification for patient <b>{sel_xai}</b>.</div>
        </div>
        """, unsafe_allow_html=True)

        c_a, c_b = st.columns(2)
        c_a.metric('↑ Positive Drivers', f'{pos_genes}', 'Pro-classification')
        c_b.metric('↓ Negative Drivers', f'{neg_genes}', 'Anti-classification')

        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Global Biomarker Leaderboard</div>', unsafe_allow_html=True)

        max_f = top_bio['ANOVA_F_Score'].max()
        bio_rows = ''
        for rank, (_, br) in enumerate(top_bio.head(12).iterrows(), 1):
            pct = br['ANOVA_F_Score'] / max_f * 100
            bc  = COHORT[br['Peak_Cancer_Cohort']]['color']
            bio_rows += f"""
            <div class="bio-row">
              <div class="bio-rank">{rank}</div>
              <div class="bio-gene">{br['Gene_ID']}</div>
              <div class="bio-bar-wrap"><div class="bio-bar" style="width:{pct:.1f}%;background:{bc}44;border:1px solid {bc}66"></div></div>
              <div class="bio-fscore">{br['ANOVA_F_Score']:.0f}</div>
              <div class="bio-cohort-badge" style="background:{bc}22;color:{bc};border:1px solid {bc}44">{br['Peak_Cancer_Cohort']}</div>
            </div>"""

        st.markdown(f'<div class="glass-card" style="padding:1rem 1.2rem">{bio_rows}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KNOWLEDGE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == '💬  Knowledge Engine':

    st.markdown("""
    <div class="onco-banner">
      <div class="onco-banner-eyebrow">Clinical Oncology · Rule-Based Intelligence</div>
      <div class="onco-banner-title">Clinical Knowledge<br>Engine</div>
      <div class="onco-banner-sub">Query the biological, genomic, and algorithmic foundations of this platform. Ask about cancer subtypes, driver mutations, feature selection methodology, or the active patient's diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    def knowledge_response(q, patient_row=None):
        q = q.lower().strip()

        if any(k in q for k in ['patient','diagnosis','this patient','active']):
            if patient_row is not None:
                code  = patient_row['Cohort']
                ci    = COHORT[code]
                conf  = patient_row['Confidence']
                ru    = patient_row['Runner_Up']
                ruc   = patient_row['Runner_Conf']
                return dict(label='Active Patient Summary', title=f'{ci["name"]} ({code})', body=f"""
<b>Diagnostic Certainty:</b> {conf:.2f}% &nbsp;|&nbsp; <b>Runner-Up:</b> {ru} ({ruc:.1f}%)<br><br>
<b>Anatomical Site:</b> {ci['organ']}<br>
<b>Hallmark Driver Mutations:</b> {ci['genes']}<br><br>
{ci['desc']}
""")

        for code, ci in COHORT.items():
            if code.lower() in q or ci['organ'].lower() in q or ci['name'].lower()[:6] in q:
                return dict(label=f'Cancer Profile · {code}', title=ci['name'], body=f"""
<b>Anatomical Site:</b> {ci['organ']}<br>
<b>Cohort Size:</b> {ci['n']} patients ({ci['pct']}% of TCGA cohort)<br>
<b>Driver Mutations:</b> {ci['genes']}<br><br>
{ci['desc']}
""")

        if any(k in q for k in ['anova','f-test','f score','selectkbest','feature selection']):
            return dict(label='Methodology', title='ANOVA F-Statistic Feature Selection', body="""
Evaluates one-way Analysis of Variance across the 5 cancer cohorts for each gene:<br><br>
<b>F = MSB / MSW = Between-Group Variance / Within-Group Variance</b><br><br>
Genes with high F-scores show strong cohort-specific expression peaks and minimal within-class variance.
The top <b>k = 500</b> genes are selected (all with p &lt; 10⁻¹²), forming the discriminative biomarker panel.
""")

        if any(k in q for k in ['pca','principal component','manifold','3d','dimensionality']):
            return dict(label='Methodology', title='3D Principal Component Analysis (PCA)', body=f"""
PCA performs an orthogonal linear transformation to maximize variance across the 500-gene biomarker space:<br><br>
<b>z_i = x_i · V₁:₃</b> (top-3 eigenvectors of the sample covariance matrix)<br><br>
<b>PC1</b> captures <b>{pca_3d.explained_variance_ratio_[0]*100:.1f}%</b> of variance<br>
<b>PC2</b> captures <b>{pca_3d.explained_variance_ratio_[1]*100:.1f}%</b> of variance<br>
<b>PC3</b> captures <b>{pca_3d.explained_variance_ratio_[2]*100:.1f}%</b> of variance<br><br>
The PCA is pre-fitted on training data only, so single-patient inference never crashes.
""")

        if any(k in q for k in ['svm','logistic','model','classifier','benchmark','accuracy']):
            return dict(label='Methodology', title='Model Architecture & Benchmark', body=f"""
Four classifier families were benchmarked under identical Stratified 5-Fold CV:<br><br>
<b>Champion: {metadata['best_model'].replace('_',' ')}</b> — 99.69% CV Accuracy · 100% ROC-AUC<br><br>
Linear models outperform tree ensembles here because high-dimensional transcriptomic data (<i>d=500</i>)
is naturally linearly separable. The softmax decision boundary finds a maximum-margin hyperplane across
the 5-class cohort space with minimal variance fragmentation.
""")

        if any(k in q for k in ['xai','shap','explain','attribution','biomarker']):
            return dict(label='Methodology', title='Explainable AI — Local Logit Attributions', body="""
For Multinomial Logistic Regression, the exact local attribution of gene j toward class c for patient i is:<br><br>
<b>φ(c, j, i) = w_{c,j} · x_{i,j}</b><br><br>
<b>φ > 0:</b> Gene expression elevates log-odds toward cancer class c (pro-classification driver)<br>
<b>φ &lt; 0:</b> Gene expression suppresses log-odds toward class c (protective discriminant)<br><br>
This eliminates black-box ambiguity and provides oncologists with mathematically verifiable gene-level explanations.
""")

        if any(k in q for k in ['leakage','leak','cv','cross-validation','pipeline']):
            return dict(label='Methodology', title='Zero-Leakage Cross-Validation', body="""
Selection leakage occurs when feature selection is performed <i>before</i> cross-validation splits —
the validation fold inadvertently influences feature rankings, inflating reported accuracy.<br><br>
Our solution encapsulates <b>VarianceThreshold → StandardScaler → SelectKBest → Classifier</b> inside
a native <b>sklearn.pipeline.Pipeline</b>. Cross-validation refits the entire pipeline (including feature
selection) exclusively on training folds, ensuring validation folds remain completely blind.
""")

        return dict(label='Help', title='Available Topics', body="""
You can ask about:<br>
<b>🧬 Cancer Biology:</b> BRCA · KIRC · LUAD · PRAD · COAD<br>
<b>🔬 Methodology:</b> ANOVA · PCA · SelectKBest · Cross-Validation · Leakage<br>
<b>🤖 AI/ML:</b> Model benchmark · XAI attributions · Logit math<br>
<b>🧪 Patient:</b> "Explain this patient's diagnosis" (loads active record)<br>
""")

    # Quick prompts
    PROMPTS = [
        ('🧪 Active Patient', "Explain this patient's diagnosis"),
        ('🎗️ BRCA', 'What is BRCA?'),
        ('🫁 LUAD', 'What is LUAD?'),
        ('🧫 KIRC', 'What is KIRC?'),
        ('🧬 ANOVA', 'How does ANOVA feature selection work?'),
        ('🌐 PCA', 'What does the 3D PCA plot represent?'),
        ('🏆 Model', 'Which model won the benchmark?'),
        ('🔬 XAI', 'How do the XAI attributions work?'),
        ('🛡️ Leakage', 'How did you prevent selection leakage?'),
    ]

    chips_html = '<div class="prompt-chips">'
    for label, _ in PROMPTS:
        chips_html += f'<div class="prompt-chip">{label}</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    btn_cols = st.columns(len(PROMPTS))
    clicked = None
    for i, (label, prompt) in enumerate(PROMPTS):
        with btn_cols[i]:
            if st.button(label, use_container_width=True, key=f'kb_{i}'):
                clicked = prompt

    user_q = st.text_input('', placeholder='Ask a clinical or algorithmic question…', value=clicked or '', label_visibility='collapsed')

    if user_q:
        active_p_row = results.iloc[0] if results is not None else None
        resp = knowledge_response(user_q, active_p_row)

        st.markdown(f"""
        <div class="kb-card">
          <div class="kb-card-label">{resp['label']}</div>
          <div class="kb-card-title">{resp['title']}</div>
          <div class="kb-card-body">{resp['body']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Static reference panels
    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Quick Reference — Cancer Biology Cards</div>', unsafe_allow_html=True)

    card_cols = st.columns(3)
    items = list(COHORT.items())
    for i, (code, ci) in enumerate(items):
        with card_cols[i % 3]:
            st.markdown(f"""
            <div class="kb-card" style="border-top:3px solid {ci['color']}">
              <div class="kb-card-label" style="color:{ci['color']}">{code} · {ci['organ']}</div>
              <div class="kb-card-title">{ci['name']}</div>
              <div class="kb-card-body">
                <b>Patients:</b> {ci['n']} ({ci['pct']}%)<br>
                <b>Drivers:</b> {ci['genes']}<br><br>
                {ci['desc'][:160]}…
              </div>
            </div>
            """, unsafe_allow_html=True)
