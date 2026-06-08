# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# Provider SDKs - imported lazily to avoid crash if not installed
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import cohere
    COHERE_OK = True
except ImportError:
    COHERE_OK = False

try:
    from mistralai import Mistral
    MISTRAL_OK = True
except ImportError:
    try:
        from mistralai.client import MistralClient  # older package name
        MISTRAL_OK = True
    except ImportError:
        MISTRAL_OK = False

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements but keep header for sidebar toggle */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px; }

/* Ensure tab panel content has breathing room below sticky bar */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 1px solid #e0e4e8;
}
[data-testid="stSidebar"] * { color: #1a202c !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #2d3748 !important;
    font-size: 0.7rem !important;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1.5px dashed #cbd5e0;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e4e8;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #1a202c !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #4a5568 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Chat messages */
.chat-user {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    border-radius: 12px 12px 4px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    color: #1e40af;
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 85%;
    margin-left: auto;
}
.chat-assistant {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px 12px 12px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    color: #166534;
    font-size: 0.9rem;
    line-height: 1.7;
    max-width: 90%;
}
.chat-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.7;
    color: #4a5568;
}

/* Buttons */
.stButton > button {
    background: #ffffff;
    color: #2d3748;
    border: 1px solid #cbd5e0;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #f7fafc;
    color: #1a202c;
    border-color: #3b82f6;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #3b82f6;
    color: white;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    background: #2563eb;
}

/* Input */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    color: #1a202c !important;
    border: 1px solid #cbd5e0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
}

/* Progress bars */
.stProgress > div > div {
    background: #3b82f6 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border-radius: 8px !important;
    color: #2d3748 !important;
    font-size: 0.85rem !important;
}

/* Select box */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    color: #1a202c !important;
    border: 1px solid #cbd5e0 !important;
    border-radius: 8px !important;
}

/* ── Sticky Tab Bar ─────────────────────────
   Streamlit scrolls inside .main > div, so we
   must make that wrapper NOT clip overflow, then
   sticky works naturally on the tab-list.        */
section[data-testid="stMain"] > div:first-child {
    overflow: visible !important;
}
section[data-testid="stMain"] {
    overflow-y: auto !important;
}

.stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: rgba(248, 249, 250, 0.97);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1.5px solid #e0e4e8;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07);
    gap: 4px;
    padding: 6px 8px 0 8px;
    margin-left: -1rem;
    margin-right: -1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    color: #718096;
    border-radius: 8px 8px 0 0;
    font-size: 0.82rem;
    font-weight: 500;
    transition: background 0.15s, color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #e8f0fe;
    color: #2563eb;
}
.stTabs [aria-selected="true"] {
    background: #dbeafe !important;
    color: #1e40af !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
    border: 1px solid #cbd5e0;
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1a202c;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 2px;
}
.badge {
    background: rgba(59,130,246,0.15);
    color: #1e40af;
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* Quick prompt chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.5rem 0; }
.chip {
    background: #ffffff;
    border: 1px solid #cbd5e0;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #2d3748;
    cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { background: #dbeafe; color: #1e40af; border-color: #3b82f6; }

/* Column profile card */
.col-card {
    background: #ffffff;
    border: 1px solid #e0e4e8;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.col-name { font-weight: 600; font-size: 0.85rem; color: #1a202c; margin-bottom: 4px; }
.col-meta { font-size: 0.75rem; color: #718096; }
.fill-bar {
    height: 3px;
    background: #e0e4e8;
    border-radius: 2px;
    margin: 6px 0;
    overflow: hidden;
}
.fill-bar-inner { height: 100%; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)







# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "charts": {},
        "df": None,
        "df_raw": None,
        "file_name": "",
        "profile": None,
        "api_key": "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",
        "provider": "Groq (Free)",
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 1500,
        "temperature": 0,
        "show_data": False,
        "masked_cols": [],
        "quick_question": None,
        "switch_to_chat": False,
        "active_page": "💬 Chat",
        # Phase 2: chart theme
        "chart_theme": "Default",
        # Phase 3: scroll flag
        "_just_jumped": False,
        # Phase 1: AI chart cache
        "ai_charts_suggestions": None,
        "ai_charts_cache_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Always ensure API keys are loaded ────────────────────────────────────────
_STARTUP_KEYS = {
    "Groq (Free)":    "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",
    "Mistral (Free)": "20Q7I2NXrSo20qXeszETzS9QEN2TxI8e",
    "Gemini (Free)":  "",
    "Cohere (Free)":  "",
}
if not st.session_state.api_key and _STARTUP_KEYS.get(st.session_state.provider):
    st.session_state.api_key = _STARTUP_KEYS[st.session_state.provider]


# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# PHASE 2: Chart Theme Customization
# ─────────────────────────────────────────────
CHART_THEMES = {
    "Default":  {"template": "plotly_white",  "colors": px.colors.qualitative.Set2},
    "Ocean":    {"template": "plotly_white",  "colors": ["#0077b6","#00b4d8","#90e0ef","#0096c7","#48cae4","#023e8a","#ade8f4"]},
    "Warm":     {"template": "plotly_white",  "colors": ["#e63946","#f4a261","#e9c46a","#2a9d8f","#e76f51","#264653","#a8dadc"]},
    "Minimal":  {"template": "simple_white",  "colors": ["#555555","#888888","#aaaaaa","#333333","#666666","#999999","#bbbbbb"]},
    "Dark":     {"template": "plotly_dark",   "colors": ["#636efa","#ef553b","#00cc96","#ab63fa","#ffa15a","#19d3f3","#ff6692"]},
}

def get_theme():
    t = st.session_state.get("chart_theme", "Default")
    return CHART_THEMES.get(t, CHART_THEMES["Default"])

# Data Profiling
# ─────────────────────────────────────────────
def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n_missing = int(s.isna().sum())
        n_unique = int(non_null.nunique())
        fill_pct = round((len(non_null) / len(s)) * 100, 1) if len(s) > 0 else 0

        info = {
            "dtype": str(s.dtype),
            "total": len(s),
            "missing": n_missing,
            "missing_pct": round((n_missing / len(s)) * 100, 1) if len(s) > 0 else 0,
            "unique": n_unique,
            "fill_pct": fill_pct,
        }

        if pd.api.types.is_numeric_dtype(s):
            info["is_numeric"] = True
            info["min"] = float(non_null.min()) if len(non_null) > 0 else None
            info["max"] = float(non_null.max()) if len(non_null) > 0 else None
            info["mean"] = round(float(non_null.mean()), 4) if len(non_null) > 0 else None
            info["median"] = round(float(non_null.median()), 4) if len(non_null) > 0 else None
            info["std"] = round(float(non_null.std()), 4) if len(non_null) > 0 else None
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())
            info["outliers"] = outliers
        else:
            info["is_numeric"] = False
            vc = non_null.astype(str).value_counts()
            info["top_values"] = vc.head(5).to_dict()
            info["top_value"] = vc.index[0] if len(vc) > 0 else None
            info["top_freq"] = int(vc.iloc[0]) if len(vc) > 0 else 0

        profile[col] = info
    return profile


import hashlib
import random

# ── Privacy Masking Helpers ──────────────────────────────────────────────────

_FAKE_FIRST = [
    "Alex","Blake","Casey","Dana","Drew","Emery","Finley","Gray","Harper","Indigo",
    "Jamie","Kai","Lane","Morgan","Nico","Parker","Quinn","Reese","Sage","Taylor",
    "Avery","Brett","Cameron","Devon","Ellis","Frankie","Glen","Hayden","Jordan","Kerry",
    "Leslie","Marley","Noel","Oakley","Perry","Remy","Skyler","Tobi","Val","Wren",
]
_FAKE_LAST = [
    "Adams","Brown","Clark","Davis","Evans","Foster","Green","Harris","Irving","Jones",
    "Kumar","Lewis","Moore","Nelson","Owen","Patel","Quinn","Rivera","Smith","Taylor",
    "Unger","Vance","Walsh","Xavier","Yang","Zhang","Allen","Bell","Cook","Dean",
    "Ellison","Flynn","Grant","Hayes","Ingram","James","Kline","Lynch","Marsh","Nash",
]

def _deterministic_seed(value: str) -> int:
    """Convert any string to a stable integer seed using SHA-256."""
    return int(hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(), 16) % (2**32)

def _looks_like_name(col: str, sample_values) -> bool:
    """Heuristic: does this column likely contain person names?"""
    col_lower = col.lower()
    name_hints = ["name", "person", "contact", "employee", "staff", "user",
                  "customer", "client", "agent", "vendor", "shipper", "consignee",
                  "operator", "owner", "driver", "captain", "master"]
    if any(h in col_lower for h in name_hints):
        return True
    # Check if values look like "Firstname Lastname" patterns
    try:
        sample = [str(v) for v in sample_values if pd.notna(v)][:20]
        word_counts = [len(s.split()) for s in sample if s and s != "nan"]
        if word_counts and 1 <= np.mean(word_counts) <= 4:
            alpha_ratio = np.mean([sum(c.isalpha() or c == " " for c in s) / max(len(s), 1) for s in sample])
            return alpha_ratio > 0.75
    except Exception:
        pass
    return False

def _looks_like_id(col: str) -> bool:
    """Heuristic: does this column likely contain IDs / reference numbers?"""
    col_lower = col.lower()
    id_hints = ["id", "no", "num", "number", "ref", "code", "key", "serial",
                "passport", "national", "ssn", "nric", "pan", "uid", "uuid",
                "imo", "mmsi", "license", "licence", "registration", "account",
                "card", "phone", "mobile", "tel", "email", "mail", "address",
                "ip", "mac", "iban", "vat", "tax"]
    return any(h in col_lower for h in id_hints)

def _build_sequential_map(series: pd.Series, label: str) -> dict:
    """
    Build a mapping: unique real value -> "Label N"
    Order is determined by first appearance in the series.
    e.g. sudhir -> User 1, priya -> User 2, ravi -> User 3
    """
    seen = {}
    counter = 1
    for v in series:
        if pd.isna(v) or str(v).strip() == "":
            continue
        key = str(v)
        if key not in seen:
            seen[key] = f"{label} {counter}"
            counter += 1
    return seen


def apply_masking(df: pd.DataFrame, masked_cols: list) -> pd.DataFrame:
    """
    Return a copy of df with selected columns privacy-masked using
    sequential numbering. Same real value always gets the same number.

    Strategy per column type:
    - Text / name  ->  Label N  (e.g. sudhir -> User 1, priya -> User 2)
    - Numeric ID   ->  Label N  (e.g. 1042 -> ID 1, 2087 -> ID 2)
    - Numeric value->  sequential rank number kept consistent
    - Datetime     ->  Date 1, Date 2 ... (unique dates numbered by order)
    """
    if not masked_cols:
        return df

    df_masked = df.copy()

    for col in masked_cols:
        if col not in df_masked.columns:
            continue

        series = df_masked[col]

        # ── Datetime columns ─────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            col_label = "Date"
            val_map = _build_sequential_map(series.astype(str), col_label)
            df_masked[col] = series.astype(str).map(
                lambda v: val_map.get(v, v) if pd.notna(v) and str(v).strip() not in ("", "NaT") else v
            )

        # ── Numeric columns ──────────────────────────────────────────
        elif pd.api.types.is_numeric_dtype(series):
            # Pick a label based on column name
            col_lower = col.lower()
            if any(h in col_lower for h in ["name", "person", "user", "customer",
                                             "client", "employee", "staff", "agent"]):
                col_label = "User"
            elif any(h in col_lower for h in ["id", "no", "num", "number", "ref",
                                               "code", "key", "serial", "account"]):
                col_label = "ID"
            else:
                col_label = col.strip().title() or "Value"

            val_map = _build_sequential_map(series.astype(str), col_label)
            df_masked[col] = series.astype(str).map(
                lambda v: val_map.get(v, v) if v not in ("nan", "NaN", "") else np.nan
            )

        # ── Text / object columns ────────────────────────────────────
        else:
            # Determine a friendly label from the column name
            col_lower = col.lower()
            if any(h in col_lower for h in ["name", "person", "user", "customer",
                                             "client", "employee", "staff", "agent",
                                             "operator", "owner", "driver", "shipper",
                                             "consignee", "captain", "master", "vendor"]):
                col_label = "User"
            elif any(h in col_lower for h in ["id", "no", "num", "number", "ref",
                                               "code", "key", "serial", "passport",
                                               "national", "account", "card", "imo",
                                               "mmsi", "license", "registration"]):
                col_label = "ID"
            elif any(h in col_lower for h in ["email", "mail"]):
                col_label = "Email"
            elif any(h in col_lower for h in ["phone", "mobile", "tel", "contact"]):
                col_label = "Phone"
            elif any(h in col_lower for h in ["address", "location", "city", "town",
                                               "port", "place", "country"]):
                col_label = "Location"
            elif any(h in col_lower for h in ["vessel", "ship", "boat", "imo"]):
                col_label = "Vessel"
            elif any(h in col_lower for h in ["company", "org", "organisation",
                                               "organization", "firm", "corp"]):
                col_label = "Company"
            else:
                # Use the column name itself as the label (title-cased, first word)
                col_label = col.strip().split()[0].title() if col.strip() else "Value"

            val_map = _build_sequential_map(series, col_label)
            df_masked[col] = series.map(
                lambda v: val_map.get(str(v), v)
                if pd.notna(v) and str(v).strip() != ""
                else v
            )

    return df_masked




# ─────────────────────────────────────────────
# Multi-file loader helper
# ─────────────────────────────────────────────
def load_files(file_list: list, masked_cols: list):
    """
    Read one or more uploaded files and return (df_raw, df, file_label, errors).
    Multiple files are concatenated column-union style (outer join so no data is lost).
    Mixed CSV / Excel is supported.
    """
    dfs = []
    names = []
    errors = []

    for f in file_list:
        try:
            if f.name.lower().endswith(".csv"):
                _d = pd.read_csv(f)
            else:
                _d = pd.read_excel(f)
            dfs.append(_d)
            names.append(f.name)
        except Exception as e:
            errors.append(f"❌ {f.name}: {e}")

    if not dfs:
        return None, None, "", errors

    if len(dfs) == 1:
        df_raw = dfs[0]
        label  = names[0]
    else:
        # Add a source column so the user can filter by file later
        for i, (d, n) in enumerate(zip(dfs, names)):
            d["_source_file"] = n
        # Outer join — keeps all columns, fills missing with NaN
        df_raw = pd.concat(dfs, ignore_index=True, sort=False)
        label  = f"{len(dfs)} files merged ({', '.join(names)})"

    df = apply_masking(df_raw, masked_cols)
    return df_raw, df, label, errors


def build_context(df: pd.DataFrame, profile: dict) -> str:
    cols = list(df.columns)
    # Cap to 5 sample rows and 20 columns to stay within token budgets
    sample = df.head(5).astype(str).to_dict(orient="records")
    display_cols = cols[:20]

    col_details = []
    for col in display_cols:
        p = profile[col]
        if p["is_numeric"]:
            col_details.append(
                f"  • {col} [numeric]: range {p['min']}–{p['max']}, mean={p['mean']}, std={p['std']}"
            )
        else:
            top = list(p["top_values"].keys())[:3]
            col_details.append(
                f"  • {col} [text]: {p['unique']} unique, top: {top}"
            )

    truncation_note = f"\n  (+ {len(cols) - 20} more columns)" if len(cols) > 20 else ""

    return f"""
DATASET: {st.session_state.file_name}
Rows: {len(df):,} | Columns: {len(cols)}

COLUMN PROFILES:
{chr(10).join(col_details)}{truncation_note}

SAMPLE DATA (5 rows):
{json.dumps(sample, default=str)}
""".strip()




# ─────────────────────────────────────────────
# Fleet / Vessel Analytics Engine  (Wide-Format)
# ─────────────────────────────────────────────

import re as _re
import datetime as _dt

def detect_vessel_dataset(df: pd.DataFrame) -> dict:
    """Auto-detect column roles for wide-format vessel deployment data."""
    result = {
        "vessel_col": None,
        "hostname_col": None,
        "imo_col": None,
        "status_col": None,
        "common_status_col": None,
        "hotfix_cols": [],      # list of (col_name, planned_date, short_label)
        "release_col": None,    # base/first release column
        "date_col": None,       # first date-like column
        "delay_col": None,      # delay column if present
    }
    for col in df.columns:
        cl = col.lower()
        if result["vessel_col"] is None and "vessel" in cl and "name" in cl:
            result["vessel_col"] = col
        if result["hostname_col"] is None and "host" in cl:
            result["hostname_col"] = col
        if result["imo_col"] is None and cl.startswith("imo"):
            result["imo_col"] = col
        if result["status_col"] is None and cl == "status":
            result["status_col"] = col
        if result["common_status_col"] is None and "common" in cl and "status" in cl:
            result["common_status_col"] = col
        if result["delay_col"] is None and "delay" in cl:
            result["delay_col"] = col
        # Hotfix / release columns: contain a date pattern dd/mm/yyyy
        m = _re.search(r'(\d{2}/\d{2}/\d{4})', col)
        if m:
            try:
                planned = pd.to_datetime(m.group(1), dayfirst=True)
                # Build short label e.g. "Base Release", "Hotfix 1", "Hotfix 8"
                hm = _re.search(r'[Hh]otfix\s*(\d+)', col)
                label = f"Hotfix {hm.group(1)}" if hm else "Base Release"
                result["hotfix_cols"].append((col, planned, label))
            except Exception:
                pass
    # Sort hotfix cols by planned date
    result["hotfix_cols"].sort(key=lambda x: x[1])
    # Fallback vessel col
    if result["vessel_col"] is None:
        cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat:
            result["vessel_col"] = cat[0]
    # Populate release_col and date_col from hotfix_cols
    if result["hotfix_cols"]:
        result["release_col"] = result["hotfix_cols"][0][0]
        result["date_col"]    = result["hotfix_cols"][0][0]
    else:
        for col in df.columns:
            cl = col.lower()
            if any(k in cl for k in ("date", "time", "release", "install")):
                result["release_col"] = col
                result["date_col"]    = col
                break
    return result


def _is_installed(val) -> bool:
    """True if cell contains an install date (not Offline, not NaN)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip().lower()
    return s not in ("offline", "nan", "", "none", "nat")


def _is_offline(val) -> bool:
    s = str(val).strip().lower()
    return s == "offline"


def build_hotfix_summary(df: pd.DataFrame, hf_cols: list) -> pd.DataFrame:
    """
    For each hotfix column compute:
    installed count, offline count, avg delay (days), max delay, success %.
    """
    rows = []
    for col, planned, label in hf_cols:
        installed_mask = df[col].apply(_is_installed)
        offline_mask   = df[col].apply(_is_offline)
        installed_count = int(installed_mask.sum())
        offline_count   = int(offline_mask.sum())
        total = len(df)

        delays = []
        for val in df.loc[installed_mask, col]:
            try:
                d = (pd.to_datetime(val) - planned).days
                delays.append(d)
            except Exception:
                pass

        rows.append({
            "Label":         label,
            "Planned Date":  planned.strftime("%d/%m/%Y"),
            "Installed":     installed_count,
            "Offline":       offline_count,
            "Total":         total,
            "Success %":     round(installed_count / total * 100, 1),
            "Avg Delay (d)": round(np.mean(delays), 1) if delays else None,
            "Max Delay (d)": int(max(delays)) if delays else None,
            "_planned_dt":   planned,
        })
    return pd.DataFrame(rows)


def compute_fleet_kpis(df: pd.DataFrame, cols: dict) -> dict:
    """Compute the 5 priority KPI cards."""
    hf_cols    = cols.get("hotfix_cols", [])
    status_col = cols.get("status_col")
    total      = len(df)

    kpis = {
        "total_vessels": total,
        "live_vessels": 0,
        "offline_vessels": 0,
        "latest_hotfix_adoption_pct": None,
        "fleet_compliance_pct": None,
    }

    if status_col and status_col in df.columns:
        kpis["live_vessels"]    = int((df[status_col].astype(str).str.lower() == "live").sum())
        kpis["offline_vessels"] = int(df.apply(
            lambda r: any(_is_offline(r[c]) for c, _, __ in hf_cols), axis=1).sum())

    if hf_cols:
        # Latest hotfix = last by planned date
        last_col, _, _ = hf_cols[-1]
        installed_latest = df[last_col].apply(_is_installed).sum()
        kpis["latest_hotfix_adoption_pct"] = round(installed_latest / total * 100, 1) if total else 0

        # Compliance: vessel has install date in EVERY hotfix column
        fully_compliant = df.apply(
            lambda r: all(_is_installed(r[c]) for c, _, __ in hf_cols), axis=1
        ).sum()
        kpis["fleet_compliance_pct"] = round(fully_compliant / total * 100, 1) if total else 0

    return kpis


# ─────────────────────────────────────────────
# Chart Detection & Generation
# ─────────────────────────────────────────────

CHART_KEYWORDS = {
    "bar":        ["bar chart", "bar graph", "bar plot", "show bar", "barchart"],
    "line":       ["line chart", "line graph", "trend", "over time", "time series", "timeline"],
    "pie":        ["pie chart", "pie graph", "donut", "proportion", "share of", "breakdown of"],
    "scatter":    ["scatter", "scatter plot", "correlation between", "relationship between", "vs "],
    "histogram":  ["histogram", "distribution of", "frequency of", "spread of"],
    "box":        ["box plot", "boxplot", "quartile", "outlier"],
    "heatmap":    ["heatmap", "heat map", "correlation matrix", "correlations"],
    "treemap":    ["treemap", "tree map"],
    "funnel":     ["funnel"],
    "area":       ["area chart", "area graph", "stacked area"],
}

def detect_chart_intent(question: str, ai_response: str = ""):
    """Detect chart intent from user question AND AI response text."""
    q = (question + " " + ai_response).lower()

    # Specific chart types first
    for chart_type, keywords in CHART_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return chart_type

    # Broad visual intent keywords — user question only (avoid false positives from AI text)
    uq = question.lower()
    broad = [
        "show", "plot", "visualize", "visualise", "chart", "graph", "display",
        "draw", "render", "see", "view", "analyse", "analyze", "compare",
        "breakdown", "break down", "top ", "bottom ", "rank", "frequency",
        "count", "how many", "distribution", "spread", "pattern", "insight",
        "summarize", "overview", "by ", "per ", "across",
    ]
    if any(w in uq for w in broad):
        return "auto"

    # If AI response mentions a chart is coming, auto-generate
    ai_chart_hints = [
        "chart rendering", "chart below", "chart will show", "rendering below",
        "graph below", "visualization below", "visual below", "plot below",
        "bar chart", "line chart", "pie chart", "scatter", "histogram",
    ]
    if any(h in ai_response.lower() for h in ai_chart_hints):
        return "auto"

    return None


def find_best_columns(question: str, df: pd.DataFrame, chart_type: str):
    """Smart column picker from the question text."""
    q = question.lower()
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Find columns mentioned by name in the question
    mentioned = [c for c in all_cols if c.lower() in q]

    x_col = y_col = color_col = None

    if chart_type in ("bar", "pie", "treemap", "funnel"):
        x_col = mentioned[0] if mentioned else (cat_cols[0] if cat_cols else all_cols[0])
        y_col = mentioned[1] if len(mentioned) > 1 else (num_cols[0] if num_cols else None)

    elif chart_type in ("line", "area"):
        # Try to find a date column
        date_cols = [c for c in all_cols if "date" in c.lower() or "time" in c.lower() or "month" in c.lower() or "year" in c.lower()]
        x_col = date_cols[0] if date_cols else (mentioned[0] if mentioned else all_cols[0])
        y_col = mentioned[1] if len(mentioned) > 1 else (num_cols[0] if num_cols else None)

    elif chart_type in ("scatter",):
        x_col = mentioned[0] if mentioned else (num_cols[0] if num_cols else all_cols[0])
        y_col = mentioned[1] if len(mentioned) > 1 else (num_cols[1] if len(num_cols) > 1 else num_cols[0] if num_cols else None)
        color_col = mentioned[2] if len(mentioned) > 2 else (cat_cols[0] if cat_cols else None)

    elif chart_type in ("histogram", "box"):
        x_col = mentioned[0] if mentioned else (num_cols[0] if num_cols else all_cols[0])

    elif chart_type == "heatmap":
        # Use all numeric cols
        x_col = None

    else:  # auto
        x_col = mentioned[0] if mentioned else (cat_cols[0] if cat_cols else all_cols[0])
        y_col = mentioned[1] if len(mentioned) > 1 else (num_cols[0] if num_cols else None)

    return x_col, y_col, color_col


def generate_chart(question: str, df: pd.DataFrame, chart_type: str):
    """Generate a Plotly figure based on question intent and data."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    x_col, y_col, color_col = find_best_columns(question, df, chart_type)

    TEMPLATE = "plotly_white"
    COLORS = px.colors.qualitative.Set2

    try:
        if chart_type == "bar":
            if x_col and y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                grp = df.groupby(df[x_col].astype(str))[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(20)
                fig = px.bar(grp, x=x_col, y=y_col, color=x_col,
                             color_discrete_sequence=COLORS,
                             title=f"{y_col} by {x_col}", template=TEMPLATE)
            elif x_col:
                vc = df[x_col].fillna("(blank)").astype(str).value_counts().head(20).reset_index()
                vc.columns = [x_col, "Count"]
                fig = px.bar(vc, x=x_col, y="Count", color=x_col,
                             color_discrete_sequence=COLORS,
                             title=f"Top values in {x_col}", template=TEMPLATE)
            else:
                return None
            fig.update_xaxes(tickangle=-35)

        elif chart_type == "line":
            if x_col and y_col:
                plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                fig = px.line(plot_df, x=x_col, y=y_col,
                              title=f"{y_col} over {x_col}", template=TEMPLATE,
                              color_discrete_sequence=["#3b82f6"])
                fig.update_traces(line_width=2.5)
            else:
                return None

        elif chart_type == "area":
            if x_col and y_col:
                plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                fig = px.area(plot_df, x=x_col, y=y_col,
                              title=f"{y_col} over {x_col}", template=TEMPLATE,
                              color_discrete_sequence=["#3b82f6"])
            else:
                return None

        elif chart_type == "pie":
            if x_col:
                vc = df[x_col].fillna("(blank)").astype(str).value_counts().head(12).reset_index()
                vc.columns = [x_col, "Count"]
                fig = px.pie(vc, names=x_col, values="Count",
                             title=f"Distribution of {x_col}", template=TEMPLATE,
                             color_discrete_sequence=COLORS, hole=0.35)
            else:
                return None

        elif chart_type == "scatter":
            if x_col and y_col:
                scatter_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().sample(min(2000, len(df)))
                fig = px.scatter(scatter_df, x=x_col, y=y_col,
                                 color=color_col,
                                 title=f"{x_col} vs {y_col}", template=TEMPLATE,
                                 opacity=0.7, color_discrete_sequence=COLORS)
            else:
                return None

        elif chart_type == "histogram":
            if x_col and pd.api.types.is_numeric_dtype(df[x_col]):
                fig = px.histogram(df, x=x_col, nbins=40,
                                   title=f"Distribution of {x_col}", template=TEMPLATE,
                                   color_discrete_sequence=["#3b82f6"])
                fig.update_layout(bargap=0.05)
            else:
                return None

        elif chart_type == "box":
            if x_col:
                if cat_cols and x_col in num_cols:
                    fig = px.box(df, x=cat_cols[0], y=x_col,
                                 color=cat_cols[0],
                                 title=f"{x_col} by {cat_cols[0]}", template=TEMPLATE,
                                 color_discrete_sequence=COLORS)
                else:
                    fig = px.box(df, y=x_col,
                                 title=f"Box Plot — {x_col}", template=TEMPLATE,
                                 color_discrete_sequence=["#3b82f6"])
            else:
                return None

        elif chart_type == "heatmap":
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().round(2)
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale="RdBu", zmid=0,
                    text=corr.values.round(2), texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig.update_layout(title="Correlation Heatmap", template=TEMPLATE,
                                  height=max(350, len(num_cols) * 45))
            else:
                return None

        elif chart_type == "treemap":
            if x_col:
                vc = df[x_col].fillna("(blank)").astype(str).value_counts().head(20).reset_index()
                vc.columns = [x_col, "Count"]
                fig = px.treemap(vc, path=[x_col], values="Count",
                                 title=f"Treemap of {x_col}", template=TEMPLATE,
                                 color="Count", color_continuous_scale="Blues")
            else:
                return None

        elif chart_type == "funnel":
            if x_col:
                vc = df[x_col].fillna("(blank)").astype(str).value_counts().head(15).reset_index()
                vc.columns = [x_col, "Count"]
                fig = px.funnel(vc, x="Count", y=x_col,
                                title=f"Funnel — {x_col}", template=TEMPLATE)
            else:
                return None

        else:  # auto — smart default
            if cat_cols and num_cols:
                grp = df.groupby(df[cat_cols[0]].astype(str))[num_cols[0]].sum().reset_index().sort_values(num_cols[0], ascending=False).head(15)
                fig = px.bar(grp, x=cat_cols[0], y=num_cols[0], color=cat_cols[0],
                             color_discrete_sequence=COLORS,
                             title=f"{num_cols[0]} by {cat_cols[0]}", template=TEMPLATE)
                fig.update_xaxes(tickangle=-35)
            elif num_cols:
                fig = px.histogram(df, x=num_cols[0], nbins=40, template=TEMPLATE,
                                   color_discrete_sequence=["#3b82f6"],
                                   title=f"Distribution of {num_cols[0]}")
            else:
                return None

        fig.update_layout(
            margin=dict(t=45, b=25, l=25, r=25),
            height=400,
            font=dict(family="DM Sans, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    except Exception as e:
        return None

# ─────────────────────────────────────────────
# Gemini API Call
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite Data Analyst AI specialising in vessel fleet management, deployment analytics, hotfix rollouts, and maritime operations.

CRITICAL RULE — NEVER write code of any kind (no Python, no matplotlib, no seaborn, no plt, no pandas code blocks). Charts are rendered automatically by the application. Your job is to provide analysis text only.

When analyzing data:
1. Lead with the KEY insight or answer
2. Support with specific numbers, percentages, and counts from the data
3. Identify anomalies, outliers, and patterns proactively
4. Provide actionable business recommendations
5. Use structured markdown (headers, bullets, bold for key terms)
6. Suggest follow-up analyses when relevant
7. Flag data quality issues when present

VESSEL / FLEET SPECIFIC GUIDANCE:
- Interpret IMO numbers, vessel names, hotfix versions, deployment statuses correctly
- "Live" / "Online" / "Active" / "Installed" = successful deployment
- "Offline" / "Failed" / "Missing" = unsuccessful deployment
- Hotfix/Release columns contain version identifiers (e.g. Hotfix 7, Hotfix 8, v1.2.3)
- Common Status = the vessel's overall fleet health classification
- Deployment delay = days between planned release date and actual install date

CHATBOT QUESTIONS YOU CAN ANSWER:
- "Show status for IMO_0123" — look up by IMO column
- "Which vessels are offline?" — filter Status column
- "How many vessels installed Hotfix 8?" — filter by release + status
- "Show vessels missing Hotfix 7" — find vessels not updated
- "What is the fleet compliance rate?" — % of vessels fully up to date
- "Which vessel has the highest deployment delay?" — max delay column
- "List vessels that missed more than 3 updates" — count missing releases
- "Show update history for VESSEL_0050" — filter by vessel name
- "Compare Hotfix 7 and Hotfix 8 adoption" — group by release
- "Show top 10 non-compliant vessels" — rank by missed updates

When a user asks for a chart/graph/plot/visualization:
- Acknowledge that the chart is being rendered below your response
- Describe what the chart will show and the key insight to look for
- Do NOT write any code

Be concise yet comprehensive. Always ground answers in the actual dataset provided."""


# ─────────────────────────────────────────────
# Provider Configs
# ─────────────────────────────────────────────
PROVIDERS = {
    "Gemini (Free)": {
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-05-20"],
        "placeholder": "AIza...",
        "help": "Free at aistudio.google.com — no card needed",
        "link": "https://aistudio.google.com",
    },
    "Groq (Free)": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "placeholder": "gsk_...",
        "help": "Free at console.groq.com — no card needed",
        "link": "https://console.groq.com",
    },
    "Cohere (Free)": {
        "models": ["command-r-plus-08-2024", "command-r-08-2024", "command-r7b-12-2024"],
        "placeholder": "...",
        "help": "Free tier at dashboard.cohere.com",
        "link": "https://dashboard.cohere.com",
    },
    "Mistral (Free)": {
        "models": ["mistral-small-latest", "open-mistral-7b", "open-mixtral-8x7b"],
        "placeholder": "...",
        "help": "Free tier at console.mistral.ai",
        "link": "https://console.mistral.ai",
    },
}


def call_ai_streaming(messages: list, context: str):
    api_key = st.session_state.api_key
    provider = st.session_state.provider
    model = st.session_state.model
    system = f"{SYSTEM_PROMPT}\n\n{context}"

    if not api_key:
        cfg = PROVIDERS.get(provider, {})
        yield f"⚠️ Please enter your {provider} API key in the sidebar. Get one free at {cfg.get('link', '')}"
        return

    try:
        # ── Gemini ──────────────────────────────
        if provider == "Gemini (Free)":
            if not GEMINI_OK:
                yield "❌ Run: pip install google-generativeai"
                return
            genai.configure(api_key=api_key)
            gmodel = genai.GenerativeModel(
                model_name=model,
                system_instruction=system,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=st.session_state.max_tokens,
                    temperature=0.1,
                )
            )
            history = []
            for m in messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            chat = gmodel.start_chat(history=history)
            response = chat.send_message(messages[-1]["content"], stream=True)
            for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except Exception:
                    # Chunk may have no text (e.g. finish_reason=MAX_TOKENS) — skip silently
                    pass

        # ── Groq ────────────────────────────────
        elif provider == "Groq (Free)":
            if not GROQ_OK:
                yield "❌ Run: pip install groq"
                return
            client = Groq(api_key=api_key)
            api_messages = [{"role": "system", "content": system}]
            api_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=st.session_state.max_tokens,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text

        # ── Cohere ──────────────────────────────
        elif provider == "Cohere (Free)":
            if not COHERE_OK:
                yield "❌ Run: pip install cohere"
                return
            co = cohere.ClientV2(api_key=api_key)
            api_messages = [{"role": "system", "content": system}]
            api_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
            stream = co.chat_stream(
                model=model,
                messages=api_messages,
                max_tokens=st.session_state.max_tokens,
            )
            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "message"):
                    txt = ""
                    try:
                        txt = event.delta.message.content[0].text
                    except Exception:
                        pass
                    if txt:
                        yield txt

        # ── Mistral ─────────────────────────────
        elif provider == "Mistral (Free)":
            # Use requests directly — avoids SDK version issues entirely
            import requests as _req
            api_messages = [{"role": "system", "content": system}]
            api_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
            resp = _req.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": api_messages,
                    "max_tokens": st.session_state.max_tokens,
                    "temperature": 0.1,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )
            if resp.status_code == 401:
                yield "❌ Invalid Mistral API key. Get one free at console.mistral.ai"
                return
            if resp.status_code != 200:
                yield f"❌ Mistral error {resp.status_code}: {resp.text[:200]}"
                return
            import json as _json
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(line)
                    txt = chunk["choices"][0]["delta"].get("content", "") or ""
                    if txt:
                        yield txt
                except Exception:
                    continue

        else:
            yield f"❌ Unknown provider: {provider}"

    except Exception as e:
        err = str(e)
        if "invalid" in err.lower() or "401" in err or "403" in err or "API_KEY" in err:
            yield f"❌ Invalid API key for {provider}. Please check the sidebar."
        elif "quota" in err.lower() or "429" in err or "rate" in err.lower():
            yield "⚠️ Rate limit reached. Please wait a moment and try again."
        elif "token" in err.lower() or "context" in err.lower() or "length" in err.lower() or "413" in err or "too large" in err.lower() or "too long" in err.lower():
            yield "⚠️ The request was too large for the model's context window. Try clearing the chat history (sidebar → Clear Chat) or uploading a smaller dataset."
        else:
            yield f"❌ Error: {err}"



# ─────────────────────────────────────────────
# PHASE 1: AI Auto-Suggest Charts
# ─────────────────────────────────────────────
import json as _json_ac

def ai_suggest_charts(df: pd.DataFrame, profile: dict) -> list:
    """
    Calls the active AI provider and asks it to recommend 5 charts
    for this dataset. Returns a list of chart config dicts.
    Each dict has: type, x, y, title, insight, color_by (optional)
    """
    cols = list(df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_summary = []
    for col in cols[:25]:
        p = profile[col]
        if p["is_numeric"]:
            col_summary.append(f"{col} [numeric, range {p['min']}–{p['max']}]")
        else:
            top = list(p["top_values"].keys())[:3]
            col_summary.append(f"{col} [text, {p['unique']} unique, top: {top}]")

    sample_rows = df.head(3).astype(str).to_dict(orient="records")

    prompt = f"""You are a data visualization expert. Given this dataset schema, suggest exactly 5 chart configs.

Dataset: {st.session_state.file_name}
Rows: {len(df):,}
Columns: {', '.join(cols[:25])}

Column details:
{chr(10).join(col_summary)}

Sample data (3 rows):
{_json_ac.dumps(sample_rows, default=str)[:1500]}

Respond ONLY with a valid JSON array. No markdown, no explanation, no code fences. Just the raw JSON array.

Each element must have exactly these keys:
- "type": one of bar, line, scatter, pie, histogram, box, heatmap, area
- "x": column name for X axis (must exist in columns above)  
- "y": column name for Y axis (must exist in columns above, must be numeric for most types), or null for histogram/pie
- "title": short chart title string
- "insight": one sentence describing what this chart will reveal
- "color_by": a categorical column name to color by, or null

Return only charts that make sense for the actual data. Use the most insightful combinations.
"""

    api_key = st.session_state.api_key
    provider = st.session_state.provider
    model = st.session_state.model

    if not api_key:
        return []

    raw_text = ""
    try:
        if provider == "Groq (Free)":
            if not GROQ_OK:
                return []
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2,
            )
            raw_text = resp.choices[0].message.content or ""

        elif provider == "Gemini (Free)":
            if not GEMINI_OK:
                return []
            import google.generativeai as _genai
            _genai.configure(api_key=api_key)
            gm = _genai.GenerativeModel(
                model_name=model,
                generation_config=_genai.types.GenerationConfig(max_output_tokens=800, temperature=0.2),
            )
            resp = gm.generate_content(prompt)
            raw_text = resp.text or ""

        elif provider == "Mistral (Free)":
            import requests as _req
            resp = _req.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 800, "temperature": 0.2},
                timeout=30,
            )
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"] or ""

        elif provider == "Cohere (Free)":
            if not COHERE_OK:
                return []
            co = cohere.ClientV2(api_key=api_key)
            resp = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            raw_text = resp.message.content[0].text or ""

        # Strip markdown code fences if present
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip().strip("`").strip()

        suggestions = _json_ac.loads(raw_text)
        if isinstance(suggestions, list):
            # Validate each suggestion has required keys and columns exist
            valid = []
            for s in suggestions:
                if not isinstance(s, dict):
                    continue
                if "type" not in s or "title" not in s:
                    continue
                x_ok = s.get("x") is None or s.get("x") in df.columns
                y_ok = s.get("y") is None or s.get("y") in df.columns
                if x_ok and y_ok:
                    valid.append(s)
            return valid[:5]
        return []

    except Exception:
        return []


@st.cache_data(show_spinner=False)
def ai_suggest_charts_cached(file_name: str, col_signature: str, provider: str, model: str) -> list:
    """
    Wrapper with cache key based on file + columns + provider/model.
    We pass df and profile through session state inside this call.
    """
    df = st.session_state.df
    profile = st.session_state.profile
    if df is None or profile is None:
        return []
    return ai_suggest_charts(df, profile)


def render_ai_chart(cfg: dict, df: pd.DataFrame, chart_idx: int):
    """Render a single AI-suggested chart from its config dict."""
    chart_type = cfg.get("type", "bar")
    x = cfg.get("x")
    y = cfg.get("y")
    title = cfg.get("title", "Chart")
    insight = cfg.get("insight", "")
    color_by = cfg.get("color_by")

    theme = get_theme()
    tpl = theme["template"]
    colors = theme["colors"]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Validate color_by
    if color_by and color_by not in df.columns:
        color_by = None

    try:
        fig = None

        if chart_type == "bar":
            if x and y and y in df.columns and x in df.columns:
                if pd.api.types.is_numeric_dtype(df[y]):
                    grp = (df.groupby(df[x].fillna("(blank)").astype(str))[y]
                           .sum().sort_values(ascending=False).head(20).reset_index())
                    grp.columns = [x, y]
                    fig = px.bar(grp, x=x, y=y, color=x if not color_by else None,
                                 title=title, template=tpl,
                                 color_discrete_sequence=colors, text_auto=True)
                    fig.update_xaxes(tickangle=-35)
            elif x and x in df.columns:
                vc = df[x].fillna("(blank)").astype(str).value_counts().head(20).reset_index()
                vc.columns = [x, "Count"]
                fig = px.bar(vc, x=x, y="Count", title=title, template=tpl,
                             color_discrete_sequence=colors, text_auto=True)
                fig.update_xaxes(tickangle=-35)

        elif chart_type == "line":
            if x and y and x in df.columns and y in df.columns:
                plot_df = df[[x, y] + ([color_by] if color_by else [])].dropna().sort_values(x)
                fig = px.line(plot_df, x=x, y=y, color=color_by, title=title,
                              template=tpl, color_discrete_sequence=colors)

        elif chart_type == "scatter":
            if x and y and x in df.columns and y in df.columns:
                plot_df = df[[x, y] + ([color_by] if color_by else [])].dropna().sample(min(2000, len(df)))
                fig = px.scatter(plot_df, x=x, y=y, color=color_by, title=title,
                                 template=tpl, color_discrete_sequence=colors, opacity=0.7)

        elif chart_type == "histogram":
            col = x or (num_cols[0] if num_cols else None)
            if col and col in df.columns:
                fig = px.histogram(df, x=col, color=color_by, title=title,
                                   template=tpl, color_discrete_sequence=colors,
                                   nbins=40, marginal="rug")

        elif chart_type == "box":
            col = y or (num_cols[0] if num_cols else None)
            if col and col in df.columns:
                fig = px.box(df, x=x if x and x in cat_cols else None, y=col,
                             color=color_by or (x if x and x in cat_cols else None),
                             title=title, template=tpl, color_discrete_sequence=colors,
                             points="outliers")

        elif chart_type == "pie":
            col = x or (cat_cols[0] if cat_cols else None)
            if col and col in df.columns:
                vc = df[col].fillna("(blank)").astype(str).value_counts().head(12).reset_index()
                vc.columns = [col, "Count"]
                fig = px.pie(vc, names=col, values="Count", title=title,
                             template=tpl, color_discrete_sequence=colors, hole=0.35)

        elif chart_type == "heatmap":
            if len(num_cols) >= 2:
                corr = df[num_cols[:12]].corr().round(2)
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale="RdBu", zmid=0,
                    text=corr.values.round(2), texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig.update_layout(title=title, template=tpl,
                                  height=max(350, len(num_cols[:12]) * 45))

        elif chart_type == "area":
            if x and y and x in df.columns and y in df.columns:
                plot_df = df[[x, y] + ([color_by] if color_by else [])].dropna().sort_values(x)
                fig = px.area(plot_df, x=x, y=y, color=color_by, title=title,
                              template=tpl, color_discrete_sequence=colors)

        if fig is None:
            # Fallback: best-effort bar from top categorical column
            if cat_cols:
                vc = df[cat_cols[0]].fillna("(blank)").astype(str).value_counts().head(15).reset_index()
                vc.columns = [cat_cols[0], "Count"]
                fig = px.bar(vc, x=cat_cols[0], y="Count", title=title,
                             template=tpl, color_discrete_sequence=colors)

        if fig:
            fig.update_layout(
                margin=dict(t=50, b=30, l=25, r=25), height=380,
                font=dict(family="DM Sans, sans-serif", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"ai_chart_{chart_idx}")
            if insight:
                st.caption(f"💡 {insight}")

    except Exception as e:
        st.warning(f"Could not render chart: {e}")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    # Provider selector
    provider = st.selectbox(
        "AI Provider",
        list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider),
        help="All options have a free tier"
    )
    if provider != st.session_state.provider:
        st.session_state.provider = provider
        st.session_state.model = PROVIDERS[provider]["models"][0]
        st.session_state.api_key = ""
        st.session_state.messages = []
        st.rerun()

    cfg = PROVIDERS[st.session_state.provider]

    # API key input
    api_key = st.text_input(
        f"{st.session_state.provider} API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder=cfg["placeholder"],
        help=cfg["help"],
    )
    if api_key:
        st.session_state.api_key = api_key

    st.markdown(f"🔗 [Get free key]({cfg['link']})", unsafe_allow_html=False)

    st.markdown("---")
    st.markdown("### 🤖 Model Settings")

    model_list = cfg["models"]
    current_model = st.session_state.model if st.session_state.model in model_list else model_list[0]
    st.session_state.model = st.selectbox(
        "Model",
        model_list,
        index=model_list.index(current_model),
        help="Choose based on speed vs capability"
    )

    st.session_state.max_tokens = st.slider(
        "Max Response Tokens", 500, 4000, 1500, 100
    )

    st.markdown("---")
    st.markdown("### 📁 File Upload")

    uploaded = st.file_uploader(
        "Upload dataset(s)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload one or more Excel / CSV files. Multiple files are merged automatically.",
    )

    if uploaded:
        _sb_df_raw, _sb_df_loaded, _sb_label, _sb_errors = load_files(
            uploaded, st.session_state.masked_cols
        )
        for err in _sb_errors:
            st.error(err)

        if _sb_df_raw is not None:
            _changed = (st.session_state.file_name != _sb_label)
            st.session_state.df_raw    = _sb_df_raw
            st.session_state.file_name = _sb_label
            st.session_state.df        = _sb_df_loaded
            st.session_state.profile   = profile_dataframe(_sb_df_loaded)

            if _changed or not st.session_state.messages:
                n_files = len(uploaded)
                if n_files == 1:
                    _intro = f"✅ **{uploaded[0].name}** loaded!\n\n"
                else:
                    _intro = f"✅ **{n_files} files merged** into one dataset!\n\n"
                    _intro += "\n".join(f"  - {f.name}" for f in uploaded) + "\n\n"
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": (
                        _intro
                        + f"**{len(_sb_df_raw):,} rows** × **{len(_sb_df_raw.columns)} columns**\n"
                        + f"Columns: {', '.join(_sb_df_raw.columns.tolist()[:8])}"
                        + (f"…" if len(_sb_df_raw.columns) > 8 else "") + "\n\n"
                        + "Ask me anything about your data!"
                    ),
                }]
                # Clear AI chart cache when new files are loaded
                st.session_state.ai_charts_suggestions = None
                st.session_state.ai_charts_cache_key   = ""

            if len(uploaded) == 1:
                st.success(f"✓ {len(_sb_df_raw):,} rows loaded")
            else:
                st.success(f"✓ {len(uploaded)} files · {len(_sb_df_raw):,} rows merged")

    # ── SESSION CONTROLS ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗑 Session")
    _sc1, _sc2 = st.columns(2)
    with _sc1:
        if st.button("Clear Chat", key="sb_clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with _sc2:
        if st.button("Reset All", key="sb_reset_all", use_container_width=True):
            for k in ["messages", "df", "df_raw", "file_name", "profile", "masked_cols"]:
                st.session_state[k] = [] if k in ("messages", "masked_cols") else None
            st.rerun()

    # ── DATASET STATS ──────────────────────────────────────────────
    if st.session_state.df is not None:
        st.markdown("---")
        _df = st.session_state.df
        missing_total = int(_df.isna().sum().sum())
        mem_kb = round(_df.memory_usage(deep=True).sum() / 1024, 1)
        st.markdown(f"""
        <div style="background:#f8f9fa; border:1px solid #e0e4e8; border-radius:10px;
                    padding:0.7rem 0.9rem; font-size:0.78rem; color:#4a5568; line-height:1.8;">
          <b>Dataset Info</b><br>
          📋 Rows: <b>{len(_df):,}</b><br>
          📊 Columns: <b>{len(_df.columns)}</b><br>
          ❓ Missing cells: <b>{missing_total:,}</b><br>
          💾 Memory: <b>{mem_kb} KB</b>
        </div>
        """, unsafe_allow_html=True)



# ─────────────────────────────────────────────
# Sidebar page navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📂 Navigation")
    _pages = ["💬 Chat", "🚢 Fleet Dashboard", "🗃 Data Preview", "🔍 Column Profile", "📈 Statistics", "📊 Visual Analytics"]
    _avail = _pages if st.session_state.df is not None else ["💬 Chat"]
    for _pg in _avail:
        _is_active = st.session_state.active_page == _pg
        if st.button(
            _pg,
            key=f"nav_{_pg}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state.active_page = _pg
            st.rerun()

# ─────────────────────────────────────────────
# PHASE 2: Chart Theme in Sidebar (1 customization option)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎨 Chart Theme")
    _theme_choice = st.selectbox(
        "Visual style for all charts",
        list(CHART_THEMES.keys()),
        index=list(CHART_THEMES.keys()).index(st.session_state.get("chart_theme", "Default")),
        key="sidebar_chart_theme",
        help="Changes the color palette for all charts in the app",
    )
    if _theme_choice != st.session_state.get("chart_theme", "Default"):
        st.session_state.chart_theme = _theme_choice
        # Also clear AI chart cache so it re-renders with new theme
        st.session_state.pop("ai_charts_cache", None)
        st.rerun()

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

# Header
import streamlit.components.v1 as _components

st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">📊 AI Data Analyst — Fleet Edition</div>
    <div class="app-subtitle">Gemini · Groq · Cohere · Mistral — all free · Upload your dataset and start</div>
  </div>
  <div class="badge">4 Free Providers</div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.df

# ── Main page upload panel (shown when no file loaded) ───────────────────────
if df is None:
    st.markdown("---")
    st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem 0;">
  <div style="font-size:2.5rem;">📂</div>
  <div style="font-size:1.1rem; font-weight:600; color:#1a202c; margin-top:0.4rem;">Upload your dataset to get started</div>
  <div style="font-size:0.82rem; color:#6b7280; margin-top:4px;">Supports Excel (.xlsx, .xls) and CSV files</div>
</div>
""", unsafe_allow_html=True)

    up_col1, up_col2, up_col3 = st.columns([1, 2, 1])
    with up_col2:
        main_upload = st.file_uploader(
            "Drop one or more files here, or click to browse",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="main_uploader",
            help="Upload multiple files to merge them automatically into one dataset",
        )
        if main_upload:
            _df_raw, _df_loaded, _label, _errors = load_files(
                main_upload, st.session_state.masked_cols
            )
            for err in _errors:
                st.error(err)

            if _df_raw is not None:
                st.session_state.df_raw    = _df_raw
                st.session_state.file_name = _label
                st.session_state.df        = _df_loaded
                st.session_state.profile   = profile_dataframe(_df_loaded)

                n_files = len(main_upload)
                if n_files == 1:
                    _intro = f"✅ **{main_upload[0].name}** loaded!\n\n"
                else:
                    _intro = f"✅ **{n_files} files merged** into one dataset!\n\n"
                    _intro += "\n".join(f"  - {f.name}" for f in main_upload) + "\n\n"

                st.session_state.messages = [{
                    "role": "assistant",
                    "content": (
                        _intro
                        + f"**{len(_df_raw):,} rows** × **{len(_df_raw.columns)} columns**\n"
                        + f"Columns: {', '.join(_df_raw.columns.tolist()[:8])}"
                        + ("…" if len(_df_raw.columns) > 8 else "") + "\n\n"
                        + "Ask me anything about your data!"
                    ),
                }]
                st.session_state.ai_charts_suggestions = None
                st.session_state.ai_charts_cache_key   = ""
                st.rerun()

        # Also show provider + API key selection inline
        st.markdown("---")
        st.markdown("**⚙️ AI Provider & Key**")
        _prov_col, _key_col = st.columns(2)
        with _prov_col:
            _provider_choice = st.selectbox(
                "Provider",
                list(PROVIDERS.keys()),
                index=list(PROVIDERS.keys()).index(st.session_state.provider),
                key="main_provider",
            )
            if _provider_choice != st.session_state.provider:
                st.session_state.provider = _provider_choice
                st.session_state.model = PROVIDERS[_provider_choice]["models"][0]
                _AUTO_KEYS2 = {
                    "Groq (Free)":    "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",
                    "Mistral (Free)": "20Q7I2NXrSo20qXeszETzS9QEN2TxI8e",
                }
                st.session_state.api_key = _AUTO_KEYS2.get(_provider_choice, "")
                st.rerun()
        with _key_col:
            _KEYS = {
                "Gemini (Free)":  "",
                "Groq (Free)":    "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",
                "Cohere (Free)":  "",
                "Mistral (Free)": "20Q7I2NXrSo20qXeszETzS9QEN2TxI8e",
            }
            if not st.session_state.api_key and _KEYS.get(st.session_state.provider):
                st.session_state.api_key = _KEYS[st.session_state.provider]
            _key_input = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="Paste API key here…",
                key="main_api_key",
            )
            if _key_input:
                st.session_state.api_key = _key_input
            if st.session_state.api_key:
                st.success("✓ Key ready")
    st.stop()

# ── Single-page routing (replaces st.tabs) ────────────────────────────────────
_active = st.session_state.active_page
# Ensure page resets to Chat when no file loaded
if df is None and _active != "💬 Chat":
    st.session_state.active_page = "💬 Chat"
    _active = "💬 Chat"

# Compatibility shims so existing tab_* guards still work
tab_fleet = tab_data = tab_profile = tab_stats = tab_viz = True if df is not None else None


# ─────────────────────────────────────────────
# PHASE 3: Auto-scroll helper (quick question UX fix)
# ─────────────────────────────────────────────
def scroll_to_bottom():
    """Inject JS that scrolls Streamlit's main scroll container to the bottom."""
    st.markdown("""
        <script>
        (function() {
            var container = window.parent.document.querySelector(
                'section[data-testid="stMain"] > div:first-child'
            );
            if (container) {
                setTimeout(function() { container.scrollTop = container.scrollHeight; }, 120);
            }
        })();
        </script>
    """, unsafe_allow_html=True)

# ── Single-page routing (replaces st.tabs) ────────────────────────────────────
_active = st.session_state.active_page
# Ensure page resets to Chat when no file loaded
if df is None and _active != "💬 Chat":
    st.session_state.active_page = "💬 Chat"
    _active = "💬 Chat"

# Compatibility shims so existing tab_* guards still work
tab_fleet = tab_data = tab_profile = tab_stats = tab_viz = True if df is not None else None

# ── Chat Page (Phase 3 integrated) ────────────────────────────────
if _active == "💬 Chat":
    # Quick prompts bar
    if df is not None:
        st.markdown("**Quick Analysis:**")
        cols_q = st.columns(6)
        quick_prompts = [
            ("📋 Summary", "Give me a complete summary and profile of this dataset including key statistics for each column."),
            ("💡 Insights", "What are the 5 most important insights or patterns you can identify from this data?"),
            ("🔍 Quality", "Analyze data quality: missing values, outliers, duplicates, and data integrity issues."),
            ("📊 Top Values", "Identify the top recurring values, frequencies, and distributions across key columns."),
            ("📈 Trends", "Identify any trends, correlations, or relationships between columns in this data."),
            ("⚠️ Anomalies", "What anomalies, outliers, or unusual patterns exist in the data?"),
        ]
        for i, (label, prompt) in enumerate(quick_prompts):
            with cols_q[i % 6]:
                if st.button(label, key=f"qp_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state._just_jumped = True
                    st.rerun()

        st.markdown("---")

    # ── PHASE 3: Auto-scroll when arriving from quick question ──
    if st.session_state.get("_just_jumped"):
        scroll_to_bottom()
        st.session_state._just_jumped = False

    # Chat history
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">'
                    f'<div class="chat-label">You</div>{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.chat_message("assistant", avatar="📊"):
                    st.markdown(msg["content"])
                    if idx in st.session_state.charts:
                        th = get_theme()
                        stored_fig = st.session_state.charts[idx]
                        stored_fig.update_layout(template=th["template"])
                        st.plotly_chart(stored_fig, use_container_width=True, key=f"chart_hist_{idx}")

    # Respond to last user message if no assistant reply yet
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "user" and df is not None:
        last_question = msgs[-1]["content"]
        context = build_context(df, st.session_state.profile)
        recent_msgs = msgs[-6:] if len(msgs) > 6 else msgs
        api_msgs = [{"role": m["role"], "content": m["content"]} for m in recent_msgs]

        with st.chat_message("assistant", avatar="📊"):
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner("Analyzing..."):
                for chunk in call_ai_streaming(api_msgs, context):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            chart_type = detect_chart_intent(last_question, full_response)
            if chart_type:
                with st.spinner("Generating chart..."):
                    fig = generate_chart(last_question, df, chart_type)
                if fig:
                    th = get_theme()
                    fig.update_layout(template=th["template"])
                    chart_placeholder = st.empty()
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_new_{len(msgs)}")
                    st.session_state.charts[len(msgs)] = fig

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

    if df is None:
        st.info("👈 Upload an Excel or CSV file in the sidebar to start analyzing your data.")

    # ── PHASE 3: Quick question input (from Fleet Dashboard) ──────
    if df is not None:
        # If arriving from a quick question click, show banner
        if st.session_state.get("quick_question") and df is not None:
            st.info(f"⚡ Sending: **{st.session_state.quick_question[:80]}**…")

        user_input = st.chat_input("Ask anything about your data…")

        # PHASE 3: Consume quick_question if set (clears immediately to prevent loop)
        qq = st.session_state.get("quick_question")
        if qq:
            st.session_state.quick_question = None
            user_input = qq

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state._just_jumped = True
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# 🚢 FLEET DASHBOARD PAGE  — Wide-format vessel deployment data
# ═══════════════════════════════════════════════════════════════════════
if _active == "🚢 Fleet Dashboard" and df is not None:
  _cols   = detect_vessel_dataset(df)
  _kpis   = compute_fleet_kpis(df, _cols)
  _hf     = _cols["hotfix_cols"]          # list of (col, planned_dt, label)
  _hf_df  = build_hotfix_summary(df, _hf) if _hf else pd.DataFrame()

  vessel_col        = _cols["vessel_col"]
  imo_col           = _cols["imo_col"]
  hostname_col      = _cols["hostname_col"]
  status_col        = _cols["status_col"]
  common_status_col = _cols["common_status_col"]
  id_col            = vessel_col or imo_col

  _TPL    = "plotly_white"
  _GREEN  = "#22c55e"
  _RED    = "#ef4444"
  _BLUE   = "#3b82f6"
  _AMBER  = "#f59e0b"
  _COLORS = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316"]

  def _fl(fig, h=420):
      fig.update_layout(
          margin=dict(t=50, b=35, l=30, r=30), height=h,
          font=dict(family="DM Sans, sans-serif", size=12),
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
      )
      return fig

  # ── 5 KPI CARDS ──────────────────────────────────────────────────
  st.markdown("### 🚢 Fleet Health Dashboard — Shippalm Release 3.3.1.46")
  k1, k2, k3, k4, k5 = st.columns(5)
  k1.metric("🚢 Total Vessels",  f"{_kpis['total_vessels']:,}")
  k2.metric("🟢 Live Vessels",   f"{_kpis['live_vessels']:,}",
            delta=f"{round(_kpis['live_vessels']/_kpis['total_vessels']*100,1) if _kpis['total_vessels'] else 0}%")
  k3.metric("🔴 Vessels w/ Offline",f"{_kpis['offline_vessels']:,}",
            delta=f"-{round(_kpis['offline_vessels']/_kpis['total_vessels']*100,1) if _kpis['total_vessels'] else 0}%",
            delta_color="inverse")
  k4.metric("📦 Latest Hotfix Adoption",
            f"{_kpis['latest_hotfix_adoption_pct']}%" if _kpis['latest_hotfix_adoption_pct'] is not None else "N/A")
  k5.metric("✅ Fleet Compliance",
            f"{_kpis['fleet_compliance_pct']}%" if _kpis['fleet_compliance_pct'] is not None else "N/A",
            help="% of vessels with install date on ALL hotfixes")

  st.markdown("---")

  # ── SECTION PICKER ────────────────────────────────────────────────
  fleet_section = st.selectbox("📂 Jump to section", [
      "📋 Summary & Suggested Questions",
      "📊 Deployment & Upgrade Analysis",
      "🔍 Vessel Status Analysis",
      "⏱ Delay Analysis",
      "🏷 Common Status Analysis",
      "🔎 Vessel / IMO Lookup",
      "📈 KPI Metrics",
      "📉 Visualizations",
  ], key="fleet_section")
  st.markdown("---")

  # ════════════════════════════════════════════════════════════════
  # 0. SUMMARY & SUGGESTED QUESTIONS
  # ════════════════════════════════════════════════════════════════
  if "Summary" in fleet_section:
      st.markdown("## 📋 Dataset Summary")

      s1, s2 = st.columns([1, 1])

      with s1:
          st.markdown("### 📁 Dataset Overview")
          meta_rows = [
              ("Total Vessels",             f"{_kpis['total_vessels']:,}"),
              ("Hotfix Releases",            str(len(_hf))),
              ("Live Vessels (Status col)",  f"{_kpis['live_vessels']:,}"),
              ("Vessels with any Offline",   f"{_kpis['offline_vessels']:,}"),
              ("Fleet Compliance",           f"{_kpis['fleet_compliance_pct']}%"),
              ("Latest Hotfix Adoption",     f"{_kpis['latest_hotfix_adoption_pct']}%"),
          ]
          st.dataframe(pd.DataFrame(meta_rows, columns=["Metric", "Value"]),
                       use_container_width=True, hide_index=True)

          if status_col:
              st.markdown("### 🏷 Vessel Status Breakdown")
              sc = df[status_col].value_counts().reset_index()
              sc.columns = ["Status", "Count"]
              sc["% of Fleet"] = (sc["Count"] / len(df) * 100).round(1)
              st.dataframe(sc, use_container_width=True, hide_index=True)

      with s2:
          st.markdown("### 📦 Hotfix Deployment Summary")
          if not _hf_df.empty:
              disp = _hf_df[["Label","Planned Date","Installed","Offline","Success %","Avg Delay (d)","Max Delay (d)"]].copy()
              st.dataframe(disp, use_container_width=True, hide_index=True)

              fig_s = go.Figure()
              fig_s.add_bar(x=_hf_df["Label"], y=_hf_df["Installed"],
                            name="Installed", marker_color=_GREEN)
              fig_s.add_bar(x=_hf_df["Label"], y=_hf_df["Offline"],
                            name="Offline", marker_color=_RED)
              fig_s.add_scatter(x=_hf_df["Label"], y=_hf_df["Success %"],
                                mode="lines+markers", name="Success %",
                                yaxis="y2", line=dict(color=_BLUE, width=2.5))
              fig_s.update_layout(
                  barmode="stack",
                  yaxis2=dict(overlaying="y", side="right", title="Success %", range=[80, 102]),
                  title="Deployment Overview by Hotfix", template=_TPL,
                  xaxis_tickangle=-30,
              )
              st.plotly_chart(_fl(fig_s, h=380), use_container_width=True)

      st.markdown("---")

      # ── QUICK ACCESS QUESTION BUTTONS ────────────────────────────
      st.markdown("## 💬 Quick Analysis — Click to Ask the AI")
      st.caption("Click any button below — the question goes straight to the Chat tab and gets answered instantly.")


      # ── PHASE 3: Updated _ask() — direct submit, no queue loop ─────
      def _ask(q):
          """Send a quick question directly into chat history and switch page."""
          st.session_state.messages.append({"role": "user", "content": q})
          st.session_state.active_page = "💬 Chat"
          st.session_state._just_jumped = True   # triggers auto-scroll on Chat page
          st.session_state.quick_question = None  # clear any stale queue
          st.rerun()

      quick_groups = [
          ("📊 Deployment & Upgrade", [
              ("How many vessels installed each hotfix?",           "How many vessels successfully installed each hotfix? Show a breakdown per hotfix."),
              ("Latest hotfix adoption %",                          "What percentage of vessels have the latest Hotfix 8 installed?"),
              ("Hotfix with highest success rate",                  "Which hotfix had the highest deployment success rate?"),
              ("Hotfix with most offline vessels",                  "Which hotfix had the most offline vessels?"),
              ("Deployment trend across all hotfixes",              "What is the deployment trend across all hotfixes? Are installations improving or declining?"),
          ]),
          ("🔍 Vessel Status", [
              ("Live vs other statuses count",                      "How many vessels are currently Live versus other statuses? Give a full breakdown."),
              ("Vessels offline for multiple hotfixes",             "Which vessels have been offline across multiple hotfixes?"),
              ("Fully compliant vessels %",                         "What percentage of the fleet is fully compliant — meaning all hotfixes installed?"),
              ("Vessels that missed the most updates",              "Which vessels missed the most hotfix updates? List the top 10."),
              ("Vessels not Live (sold, AOT, etc.)",                "List all vessels that are not Live — include Vessel sold, AOT Configuration, Ready to Dispatch, etc."),
          ]),
          ("⏱ Delay Analysis", [
              ("Average deployment delay",                          "What is the average deployment delay across all hotfixes in days?"),
              ("Vessels with longest install delay",                "Which vessels took the longest to install a hotfix? Show top 10."),
              ("Hotfix with highest average delay",                 "Which hotfix release had the highest average installation delay?"),
              ("Vessels installed on release date",                 "How many vessels installed Hotfix 8 on or before the release date?"),
              ("Vessels consistently late",                         "Which vessels consistently install updates late across multiple hotfixes?"),
          ]),
          ("🏷 Common Status", [
              ("Most common vessel statuses",                       "What are the most common vessel statuses in the fleet? Show percentages."),
              ("AOT Configuration count",                           "How many vessels are in AOT Configuration status?"),
              ("Vessel sold count",                                 "How many vessels are listed as Vessel sold?"),
              ("Vessels with no status",                            "Are there any vessels with no common status assigned?"),
              ("Status distribution %",                             "What is the full distribution of vessel statuses as a percentage of the fleet?"),
          ]),
          ("📈 KPI Metrics", [
              ("Fleet compliance rate",                             "What is the current fleet compliance rate and how is it calculated?"),
              ("Overall update success rate",                       "What is the overall update success rate across all hotfixes?"),
              ("Offline rate for latest hotfix",                    "What is the offline rate for the latest hotfix (Hotfix 8)?"),
              ("Average deployment delay in days",                  "What is the average deployment delay in days across the entire fleet?"),
              ("Hotfix 8 adoption rate",                            "What is the Hotfix 8 adoption rate? How does it compare to earlier hotfixes?"),
          ]),
          ("🔎 Vessel Lookup", [
              ("Vessels missing Hotfix 8",                          "Which vessels are missing Hotfix 8 — either Offline or not installed?"),
              ("Vessels missing Hotfix 7",                          "Which vessels are missing Hotfix 7?"),
              ("Vessels with all hotfixes installed",               "List all vessels that have all 9 hotfixes successfully installed."),
              ("Compare Hotfix 7 vs Hotfix 8 adoption",            "Compare Hotfix 7 and Hotfix 8 adoption rates — how many vessels installed each?"),
              ("Top 10 non-compliant vessels",                      "Show the top 10 most non-compliant vessels — those missing the most hotfix updates."),
          ]),
      ]

      for group_name, questions in quick_groups:
          st.markdown(f"#### {group_name}")
          cols_per_row = 3
          for i in range(0, len(questions), cols_per_row):
              row_qs = questions[i:i+cols_per_row]
              btn_cols = st.columns(cols_per_row)
              for col_idx, (label, full_q) in enumerate(row_qs):
                  with btn_cols[col_idx]:
                      if st.button(
                          label,
                          key=f"qq_{group_name}_{i}_{col_idx}",
                          use_container_width=True,
                          help=full_q,
                      ):
                          _ask(full_q)
          st.markdown("")
  # ════════════════════════════════════════════════════════════════
  # 1. DEPLOYMENT & UPGRADE ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Deployment" in fleet_section:
      st.markdown("#### 📊 Deployment & Upgrade Analysis")

      if not _hf_df.empty:
          da1, da2 = st.columns(2)
          with da1:
              st.markdown("**Q1 — Vessels completing each hotfix**")
              st.dataframe(
                  _hf_df[["Label","Planned Date","Installed","Offline","Total","Success %"]],
                  use_container_width=True, hide_index=True)
              best_row = _hf_df.loc[_hf_df["Success %"].idxmax()]
              st.metric("Q3 — Highest success rate",
                        best_row["Label"], f"{best_row['Success %']}%")
          with da2:
              latest_row = _hf_df.iloc[-1]
              st.metric(f"Q2 — Latest hotfix ({latest_row['Label']}) adoption",
                        f"{latest_row['Success %']}%",
                        f"{int(latest_row['Installed'])} vessels installed")
              worst_row = _hf_df.loc[_hf_df["Offline"].idxmax()]
              st.metric("Q4 — Most offline vessels in a hotfix",
                        worst_row["Label"], f"{int(worst_row['Offline'])} offline",
                        delta_color="inverse")

          st.markdown("**Q5 — Deployment trend across all hotfixes**")
          fig = go.Figure()
          fig.add_bar(x=_hf_df["Label"], y=_hf_df["Installed"],
                      name="Installed", marker_color=_GREEN)
          fig.add_bar(x=_hf_df["Label"], y=_hf_df["Offline"],
                      name="Offline", marker_color=_RED)
          fig.add_scatter(x=_hf_df["Label"], y=_hf_df["Success %"],
                          mode="lines+markers", name="Success %",
                          yaxis="y2", line=dict(color=_BLUE, width=2.5))
          fig.update_layout(
              barmode="stack",
              yaxis2=dict(overlaying="y", side="right", title="Success %", range=[80,102]),
              title="Deployment Success Trend — All Hotfixes", template=_TPL,
              xaxis_tickangle=-30,
          )
          st.plotly_chart(_fl(fig, h=430), use_container_width=True)
      else:
          st.info("No hotfix columns detected.")

  # ════════════════════════════════════════════════════════════════
  # 2. VESSEL STATUS ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Vessel Status" in fleet_section:
      st.markdown("#### 🔍 Vessel Status Analysis")

      vs1, vs2 = st.columns(2)
      if status_col:
          sc2 = df[status_col].value_counts().reset_index()
          sc2.columns = ["Status", "Count"]
          sc2["% Fleet"] = (sc2["Count"] / len(df) * 100).round(1)
          with vs1:
              st.markdown("**Q1 — Live vs other statuses**")
              fig_pie = px.pie(sc2, names="Status", values="Count",
                  color_discrete_sequence=_COLORS, hole=0.4,
                  title="Vessel Status Distribution", template=_TPL)
              st.plotly_chart(_fl(fig_pie, h=380), use_container_width=True)
          with vs2:
              st.markdown("**Q1 — Count by status**")
              st.dataframe(sc2, use_container_width=True, hide_index=True)

      # Q2 — vessels offline in multiple hotfixes
      if _hf and id_col:
          st.markdown("**Q2 — Vessels offline across multiple hotfixes**")
          offline_counts = pd.Series(0, index=df.index)
          for col, _, __ in _hf:
              offline_counts += df[col].apply(_is_offline).astype(int)
          df["_offline_count"] = offline_counts.values
          multi_offline = df[df["_offline_count"] > 1][[id_col, "_offline_count"]].copy()
          multi_offline.columns = [id_col, "Hotfixes Offline"]
          multi_offline = multi_offline.sort_values("Hotfixes Offline", ascending=False).head(20)
          if len(multi_offline):
              fig_mo = px.bar(multi_offline, x=id_col, y="Hotfixes Offline",
                  color="Hotfixes Offline", color_continuous_scale="Reds",
                  title="Vessels Offline Across Multiple Hotfixes", template=_TPL)
              fig_mo.update_xaxes(tickangle=-45, tickfont_size=9)
              st.plotly_chart(_fl(fig_mo, h=400), use_container_width=True)
          else:
              st.success("No vessels found offline in multiple hotfixes.")
          df.drop(columns=["_offline_count"], inplace=True, errors="ignore")

      # Q3 — fully compliant
      if _hf and id_col:
          st.markdown("**Q3 — Fully compliant vessels (all hotfixes installed)**")
          fully = df.apply(lambda r: all(_is_installed(r[c]) for c, _, __ in _hf), axis=1)
          total_fully = int(fully.sum())
          pct = round(total_fully / len(df) * 100, 1)
          st.metric("Fully compliant vessels", f"{total_fully}", f"{pct}% of fleet")

      # Q4 — vessels missing most updates
      if _hf and id_col:
          st.markdown("**Q4 — Vessels that missed the most hotfixes**")
          missed = df.apply(
              lambda r: sum(not _is_installed(r[c]) for c, _, __ in _hf), axis=1)
          df["_missed"] = missed.values
          top_missed = df[[id_col, "_missed"]].copy()
          top_missed.columns = [id_col, "Hotfixes Missed"]
          top_missed = top_missed[top_missed["Hotfixes Missed"] > 0].sort_values(
              "Hotfixes Missed", ascending=False).head(20)
          if len(top_missed):
              st.dataframe(top_missed, use_container_width=True, hide_index=True)
          else:
              st.success("All vessels have all hotfixes installed.")
          df.drop(columns=["_missed"], inplace=True, errors="ignore")

  # ════════════════════════════════════════════════════════════════
  # 3. DELAY ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Delay" in fleet_section:
      st.markdown("#### ⏱ Delay Analysis")

      if not _hf:
          st.info("No hotfix columns with planned dates detected.")
      else:
          # Build a flat delay table: vessel × hotfix → delay days
          delay_rows = []
          for col, planned, label in _hf:
              for idx, row in df.iterrows():
                  val = row[col]
                  if _is_installed(val):
                      try:
                          delay = (pd.to_datetime(val) - planned).days
                          delay_rows.append({
                              "Vessel": row.get(id_col, idx),
                              "Hotfix": label,
                              "Planned": planned.strftime("%d/%m/%Y"),
                              "Installed": pd.to_datetime(val).strftime("%d/%m/%Y"),
                              "Delay (days)": delay,
                          })
                      except Exception:
                          pass

          delay_df = pd.DataFrame(delay_rows)

          if len(delay_df):
              dl1, dl2, dl3, dl4 = st.columns(4)
              avg_d = delay_df["Delay (days)"].mean()
              max_d = delay_df["Delay (days)"].max()
              on_time = int((delay_df["Delay (days)"] <= 0).sum())
              dl1.metric("Q1 — Avg Delay", f"{avg_d:.1f} days")
              dl2.metric("Max Delay", f"{int(max_d)} days")
              dl3.metric("Q4 — Installed On/Early", f"{on_time:,}")
              dl4.metric("Total Install Events", f"{len(delay_df):,}")

              # Q3 — avg delay per hotfix
              st.markdown("**Q3 — Average delay by hotfix**")
              hf_delay = delay_df.groupby("Hotfix")["Delay (days)"].agg(["mean","max","count"]).round(1).reset_index()
              hf_delay.columns = ["Hotfix", "Avg Delay (d)", "Max Delay (d)", "Installs"]
              fig_hd = px.bar(hf_delay, x="Hotfix", y="Avg Delay (d)",
                  color="Avg Delay (d)", color_continuous_scale="Oranges",
                  title="Average Installation Delay by Hotfix", template=_TPL,
                  text="Avg Delay (d)")
              fig_hd.update_traces(textposition="outside")
              st.plotly_chart(_fl(fig_hd, h=380), use_container_width=True)
              st.dataframe(hf_delay, use_container_width=True, hide_index=True)

              # Q2 — top 10 slowest vessels
              st.markdown("**Q2 — Top 10 vessels by maximum delay**")
              v_delay = delay_df.groupby("Vessel")["Delay (days)"].max().sort_values(ascending=False).head(10).reset_index()
              v_delay.columns = ["Vessel", "Max Delay (days)"]
              fig_vd = px.bar(v_delay, y="Vessel", x="Max Delay (days)", orientation="h",
                  color="Max Delay (days)", color_continuous_scale="Reds",
                  title="Top 10 Vessels by Maximum Deployment Delay", template=_TPL,
                  text="Max Delay (days)")
              fig_vd.update_traces(textposition="outside")
              fig_vd.update_yaxes(categoryorder="total ascending")
              st.plotly_chart(_fl(fig_vd, h=420), use_container_width=True)

              # Histogram
              st.markdown("**Delay distribution across all install events**")
              fig_hist = px.histogram(delay_df, x="Delay (days)", nbins=50,
                  color="Hotfix", title="Delay Distribution", template=_TPL,
                  marginal="rug")
              fig_hist.add_vline(x=avg_d, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {avg_d:.1f}d")
              st.plotly_chart(_fl(fig_hist, h=400), use_container_width=True)

              # Q5 — consistently late vessels
              st.markdown("**Q5 — Vessels consistently installing late (>0 days)**")
              late_rate = (delay_df[delay_df["Delay (days)"] > 0]
                  .groupby("Vessel").size() / delay_df.groupby("Vessel").size()
              ).dropna().sort_values(ascending=False).head(15).reset_index()
              late_rate.columns = ["Vessel", "Late Rate"]
              late_rate["Late Rate %"] = (late_rate["Late Rate"] * 100).round(1)
              st.dataframe(late_rate[["Vessel","Late Rate %"]], use_container_width=True, hide_index=True)

  # ════════════════════════════════════════════════════════════════
  # 4. COMMON STATUS ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Common Status" in fleet_section:
      st.markdown("#### 🏷 Common Status Analysis")

      cs_col = common_status_col or status_col
      if cs_col and cs_col in df.columns:
          cs1, cs2 = st.columns(2)
          cs_counts = df[cs_col].fillna("(No Status)").astype(str).value_counts().reset_index()
          cs_counts.columns = ["Status", "Count"]
          cs_counts["% Fleet"] = (cs_counts["Count"] / len(df) * 100).round(1)

          with cs1:
              st.markdown("**Q1/Q5 — Status distribution**")
              fig_cs = px.pie(cs_counts, names="Status", values="Count",
                  color_discrete_sequence=_COLORS, hole=0.45,
                  title="Common Status Distribution", template=_TPL)
              st.plotly_chart(_fl(fig_cs, h=380), use_container_width=True)
          with cs2:
              st.markdown("**Q3 — Count per category**")
              st.dataframe(cs_counts, use_container_width=True, hide_index=True)

          # Q2 vessels per status
          if id_col:
              st.markdown("**Q2 — Vessels in each status**")
              for status_val, grp in df.groupby(cs_col):
                  with st.expander(f"**{status_val}** — {len(grp)} vessel(s)"):
                      cols_show = [c for c in [id_col, imo_col, hostname_col] if c]
                      st.dataframe(grp[cols_show].reset_index(drop=True), use_container_width=True)

          # Q4 no status
          no_status = df[df[cs_col].isna() | (df[cs_col].astype(str).str.strip().isin(["","nan","None"]))]
          if len(no_status):
              st.warning(f"**Q4 — {len(no_status)} vessels have no common status assigned**")
              if id_col:
                  st.dataframe(no_status[[id_col]].reset_index(drop=True), use_container_width=True)
          else:
              st.success("Q4 — All vessels have a common status assigned.")
      else:
          st.info("No Common Status column detected.")

  # ════════════════════════════════════════════════════════════════
  # 5. VESSEL / IMO LOOKUP
  # ════════════════════════════════════════════════════════════════
  elif "Lookup" in fleet_section:
      st.markdown("#### 🔎 Vessel / IMO Lookup")

      lookup_col = id_col
      if lookup_col:
          all_ids = sorted(df[lookup_col].dropna().astype(str).unique().tolist())
          search  = st.text_input("🔍 Search vessel / IMO", placeholder="Type to filter…", key="vsl_search")
          filtered = [v for v in all_ids if search.lower() in v.lower()] if search else all_ids

          sel = st.selectbox("Select vessel", filtered, key="vsl_select") if filtered else None

          if sel:
              row = df[df[lookup_col].astype(str) == sel].iloc[0]
              st.markdown(f"### 🚢 {sel}")

              # Summary cards
              iv1, iv2, iv3, iv4 = st.columns(4)
              if status_col:
                  iv1.metric("Fleet Status", str(row.get(status_col, "—")))
              if imo_col and imo_col != lookup_col:
                  iv2.metric("IMO", str(row.get(imo_col, "—")))
              if hostname_col:
                  iv3.metric("Hostname", str(row.get(hostname_col, "—")))

              installed_hf = sum(_is_installed(row[c]) for c, _, __ in _hf)
              iv4.metric("Hotfixes Installed", f"{installed_hf}/{len(_hf)}")

              # Hotfix history table
              st.markdown("**Complete Hotfix History**")
              hist_rows = []
              for col, planned, label in _hf:
                  val = row[col]
                  if _is_installed(val):
                      try:
                          installed_dt = pd.to_datetime(val)
                          delay = (installed_dt - planned).days
                          hist_rows.append({
                              "Hotfix": label,
                              "Planned Date": planned.strftime("%d/%m/%Y"),
                              "Install Date": installed_dt.strftime("%d/%m/%Y"),
                              "Delay (days)": delay,
                              "Status": "✅ Installed",
                          })
                      except Exception:
                          hist_rows.append({"Hotfix": label, "Planned Date": planned.strftime("%d/%m/%Y"),
                                            "Install Date": str(val), "Delay (days)": "—", "Status": "✅ Installed"})
                  elif _is_offline(val):
                      hist_rows.append({"Hotfix": label, "Planned Date": planned.strftime("%d/%m/%Y"),
                                        "Install Date": "—", "Delay (days)": "—", "Status": "🔴 Offline"})
                  else:
                      hist_rows.append({"Hotfix": label, "Planned Date": planned.strftime("%d/%m/%Y"),
                                        "Install Date": "—", "Delay (days)": "—", "Status": "⬜ No Data"})

              hist_df = pd.DataFrame(hist_rows)
              st.dataframe(hist_df, use_container_width=True, hide_index=True)

          # Q3 — Compare two vessels
          st.markdown("---")
          st.markdown("**Compare two vessels side-by-side**")
          c1c, c2c = st.columns(2)
          with c1c:
              va = st.selectbox("Vessel A", all_ids, key="cmp_va")
          with c2c:
              vb = st.selectbox("Vessel B", all_ids, index=min(1, len(all_ids)-1), key="cmp_vb")
          if va and vb and _hf:
              ra = df[df[lookup_col].astype(str) == va].iloc[0]
              rb = df[df[lookup_col].astype(str) == vb].iloc[0]
              cmp_rows = []
              for col, planned, label in _hf:
                  def _fmt(val, planned=planned):
                      if _is_offline(val): return "🔴 Offline"
                      if _is_installed(val):
                          try:
                              d = (pd.to_datetime(val) - planned).days
                              return f"✅ {pd.to_datetime(val).strftime('%d/%m/%Y')} (+{d}d)"
                          except: return "✅ Installed"
                      return "⬜ —"
                  cmp_rows.append({"Hotfix": label, va: _fmt(ra[col]), vb: _fmt(rb[col])})
              st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

          # Q5 — Vessels missing a specific hotfix
          st.markdown("---")
          st.markdown("**Find vessels missing a specific hotfix**")
          hf_labels = [label for _, __, label in _hf]
          sel_hf = st.selectbox("Select hotfix", hf_labels, key="miss_hf")
          if sel_hf and _hf and id_col:
              hf_col = next((c for c, _, lbl in _hf if lbl == sel_hf), None)
              if hf_col:
                  missing_mask = df[hf_col].apply(lambda x: not _is_installed(x))
                  missing_vessels = df[missing_mask][[id_col, hf_col]].copy()
                  missing_vessels.columns = [id_col, "Status"]
                  st.markdown(f"**{len(missing_vessels)} vessels missing {sel_hf}**")
                  st.dataframe(missing_vessels.reset_index(drop=True), use_container_width=True)

  # ════════════════════════════════════════════════════════════════
  # 6. KPI METRICS
  # ════════════════════════════════════════════════════════════════
  elif "KPI" in fleet_section:
      st.markdown("#### 📈 KPI Metrics")

      kp1, kp2, kp3, kp4, kp5 = st.columns(5)
      kp1.metric("✅ Fleet Compliance",
                 f"{_kpis['fleet_compliance_pct']}%" if _kpis['fleet_compliance_pct'] is not None else "N/A",
                 help="% vessels with all hotfixes installed")
      total_installs = sum(_hf_df["Installed"]) if not _hf_df.empty else 0
      total_possible = len(df) * len(_hf) if _hf else 0
      success_rate = round(total_installs / total_possible * 100, 1) if total_possible else 0
      kp2.metric("📦 Update Success Rate", f"{success_rate}%")
      off_rate = round(_kpis["offline_vessels"] / _kpis["total_vessels"] * 100, 1) if _kpis["total_vessels"] else 0
      kp3.metric("🔴 Offline Rate", f"{off_rate}%", delta_color="inverse")
      kp4.metric("📦 Latest Hotfix Adoption",
                 f"{_kpis['latest_hotfix_adoption_pct']}%" if _kpis['latest_hotfix_adoption_pct'] is not None else "N/A")
      kp5.metric("🚢 Total Vessels", f"{_kpis['total_vessels']:,}")

      st.markdown("---")
      st.markdown("**Full KPI Table**")
      kpi_table = [
          {"KPI": "Total Vessels",               "Value": _kpis["total_vessels"],             "Unit": "vessels"},
          {"KPI": "Live Vessels",                "Value": _kpis["live_vessels"],               "Unit": "vessels"},
          {"KPI": "Vessels with Offline",        "Value": _kpis["offline_vessels"],            "Unit": "vessels"},
          {"KPI": "Fleet Compliance Rate",       "Value": _kpis["fleet_compliance_pct"],       "Unit": "%"},
          {"KPI": "Latest Hotfix Adoption",      "Value": _kpis["latest_hotfix_adoption_pct"], "Unit": "%"},
          {"KPI": "Offline Rate",                "Value": off_rate,                            "Unit": "%"},
          {"KPI": "Update Success Rate",         "Value": success_rate,                        "Unit": "%"},
          {"KPI": "Total Hotfixes",              "Value": len(_hf),                            "Unit": "releases"},
      ]
      if not _hf_df.empty:
          kpi_table.append({"KPI": "Avg Delay (across all hotfixes)",
                            "Value": round(_hf_df["Avg Delay (d)"].mean(), 1), "Unit": "days"})
      st.dataframe(pd.DataFrame(kpi_table), use_container_width=True, hide_index=True)

      st.markdown("---")
      st.markdown("**Per-Hotfix KPI Breakdown**")
      if not _hf_df.empty:
          st.dataframe(
              _hf_df[["Label","Planned Date","Installed","Offline","Success %","Avg Delay (d)","Max Delay (d)"]],
              use_container_width=True, hide_index=True)

  # ════════════════════════════════════════════════════════════════
  # 7. VISUALIZATIONS
  # ════════════════════════════════════════════════════════════════
  elif "Visualiz" in fleet_section:
      st.markdown("#### 📉 Fleet Visualizations")

      chart = st.selectbox("Choose chart", [
          "Deployment Success by Hotfix (Bar)",
          "Live vs Other Statuses (Pie)",
          "Offline Count per Hotfix (Bar)",
          "Delay Trend by Hotfix (Line)",
          "Delay Distribution (Histogram)",
          "Fleet Compliance (Donut)",
          "Top 10 Slowest Vessels (Bar)",
          "Vessel × Hotfix Heatmap",
      ], key="fleet_chart")

      if chart == "Deployment Success by Hotfix (Bar)":
          if not _hf_df.empty:
              fig = go.Figure()
              fig.add_bar(x=_hf_df["Label"], y=_hf_df["Installed"], name="Installed", marker_color=_GREEN)
              fig.add_bar(x=_hf_df["Label"], y=_hf_df["Offline"],   name="Offline",   marker_color=_RED)
              fig.add_scatter(x=_hf_df["Label"], y=_hf_df["Success %"],
                              mode="lines+markers", name="Success %",
                              yaxis="y2", line=dict(color=_BLUE, width=2.5))
              fig.update_layout(barmode="group",
                  yaxis2=dict(overlaying="y", side="right", title="Success %"),
                  title="Deployment Success by Hotfix", template=_TPL)
              st.plotly_chart(_fl(fig, h=440), use_container_width=True)

      elif chart == "Live vs Other Statuses (Pie)":
          if status_col:
              sc3 = df[status_col].value_counts().reset_index()
              sc3.columns = ["Status","Count"]
              fig = px.pie(sc3, names="Status", values="Count", hole=0.4,
                  color_discrete_sequence=_COLORS,
                  title="Vessel Status Distribution", template=_TPL)
              st.plotly_chart(_fl(fig, h=440), use_container_width=True)

      elif chart == "Offline Count per Hotfix (Bar)":
          if not _hf_df.empty:
              fig = px.bar(_hf_df, x="Label", y="Offline",
                  color="Offline", color_continuous_scale="Reds",
                  title="Offline Vessel Count per Hotfix", template=_TPL, text="Offline")
              fig.update_traces(textposition="outside")
              st.plotly_chart(_fl(fig, h=420), use_container_width=True)

      elif chart == "Delay Trend by Hotfix (Line)":
          if not _hf_df.empty:
              fig = go.Figure()
              fig.add_scatter(x=_hf_df["Label"], y=_hf_df["Avg Delay (d)"],
                  mode="lines+markers", name="Avg Delay",
                  line=dict(color=_AMBER, width=2.5),
                  marker=dict(size=8))
              fig.add_scatter(x=_hf_df["Label"], y=_hf_df["Max Delay (d)"],
                  mode="lines+markers", name="Max Delay",
                  line=dict(color=_RED, width=1.5, dash="dash"),
                  marker=dict(size=7))
              fig.update_layout(title="Average & Max Installation Delay by Hotfix",
                  yaxis_title="Days", template=_TPL)
              st.plotly_chart(_fl(fig, h=420), use_container_width=True)

      elif chart == "Delay Distribution (Histogram)":
          delay_rows2 = []
          for col, planned, label in _hf:
              for val in df[col]:
                  if _is_installed(val):
                      try:
                          d = (pd.to_datetime(val) - planned).days
                          delay_rows2.append({"Hotfix": label, "Delay (days)": d})
                      except: pass
          if delay_rows2:
              dd2 = pd.DataFrame(delay_rows2)
              fig = px.histogram(dd2, x="Delay (days)", color="Hotfix", nbins=50,
                  marginal="box", template=_TPL,
                  title="Deployment Delay Distribution (all hotfixes)")
              fig.add_vline(x=dd2["Delay (days)"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {dd2['Delay (days)'].mean():.1f}d")
              st.plotly_chart(_fl(fig, h=440), use_container_width=True)

      elif chart == "Fleet Compliance (Donut)":
          if _kpis["fleet_compliance_pct"] is not None:
              cv = _kpis["fleet_compliance_pct"]
              fig = go.Figure(go.Pie(
                  values=[cv, 100-cv], labels=["Compliant","Non-Compliant"],
                  hole=0.62, marker_colors=[_GREEN, _RED]))
              fig.update_layout(title="Fleet Compliance Rate", template=_TPL,
                  annotations=[dict(text=f"{cv}%", font_size=26, showarrow=False)])
              st.plotly_chart(_fl(fig, h=420), use_container_width=True)

      elif chart == "Top 10 Slowest Vessels (Bar)":
          delay_rows3 = []
          for col, planned, label in _hf:
              for idx, row in df.iterrows():
                  val = row[col]
                  if _is_installed(val):
                      try:
                          d = (pd.to_datetime(val) - planned).days
                          delay_rows3.append({"Vessel": row.get(id_col, idx), "Delay": d})
                      except: pass
          if delay_rows3 and id_col:
              dd3 = pd.DataFrame(delay_rows3)
              top10 = dd3.groupby("Vessel")["Delay"].mean().round(1).sort_values(
                  ascending=False).head(10).reset_index()
              top10.columns = ["Vessel","Avg Delay (days)"]
              fig = px.bar(top10, y="Vessel", x="Avg Delay (days)", orientation="h",
                  color="Avg Delay (days)", color_continuous_scale="Reds",
                  title="Top 10 Vessels by Avg Deployment Delay", template=_TPL,
                  text="Avg Delay (days)")
              fig.update_traces(textposition="outside")
              fig.update_yaxes(categoryorder="total ascending")
              st.plotly_chart(_fl(fig, h=440), use_container_width=True)

      elif chart == "Vessel × Hotfix Heatmap":
          if _hf and id_col:
              hf_labels = [lbl for _, __, lbl in _hf]
              hf_colnames = [c for c, _, __ in _hf]

              # Pick bottom 40 by compliance
              compliance = df.apply(
                  lambda r: sum(_is_installed(r[c]) for c in hf_colnames), axis=1)
              bottom_idx = compliance.nsmallest(40).index
              sub = df.loc[bottom_idx].copy()

              z = sub[hf_colnames].apply(
                  lambda col_: col_.apply(lambda x: 1 if _is_installed(x) else 0)
              ).values
              vessel_names = sub[id_col].astype(str).tolist()

              fig = go.Figure(go.Heatmap(
                  z=z,
                  x=hf_labels,
                  y=vessel_names,
                  colorscale=[[0, _RED],[1, _GREEN]],
                  text=[["Installed" if v else "Offline/Missing" for v in row_] for row_ in z],
                  texttemplate="%{text}",
                  textfont={"size": 9},
                  showscale=False,
              ))
              fig.update_layout(
                  title="Vessel × Hotfix Heatmap — 40 Least Compliant Vessels",
                  template=_TPL,
                  height=max(500, len(vessel_names) * 20),
                  margin=dict(t=55, b=30, l=140, r=20),
                  yaxis=dict(autorange="reversed"),
              )
              st.plotly_chart(fig, use_container_width=True)


# ── Data Preview Tab ─────────────────────────
  _cols = detect_vessel_dataset(df)
  _kpis = compute_fleet_kpis(df, _cols)

  vessel_col  = _cols["vessel_col"]
  imo_col     = _cols["imo_col"]
  status_col  = _cols["status_col"]
  release_col = _cols["release_col"]
  date_col    = _cols["date_col"]
  delay_col   = _cols["delay_col"]
  common_status_col = _cols["common_status_col"]
  id_col = imo_col or vessel_col

  _TPL = "plotly_white"
  _COLORS = px.colors.qualitative.Set2

  # ── 5 KPI CARDS ──────────────────────────────────────────────────
  st.markdown("### 🚢 Fleet Health Dashboard")
  kc1, kc2, kc3, kc4, kc5 = st.columns(5)

  kc1.metric("🚢 Total Vessels", f"{_kpis['total_vessels']:,}")
  kc2.metric("🟢 Live Vessels",  f"{_kpis['live_vessels']:,}",
             delta=f"{round(_kpis['live_vessels']/_kpis['total_vessels']*100,1) if _kpis['total_vessels'] else 0}%" )
  kc3.metric("🔴 Offline Vessels", f"{_kpis['offline_vessels']:,}",
             delta=f"-{round(_kpis['offline_vessels']/_kpis['total_vessels']*100,1) if _kpis['total_vessels'] else 0}%",
             delta_color="inverse")
  kc4.metric("📦 Latest Hotfix Adoption",
             f"{_kpis['latest_hotfix_adoption_pct']}%" if _kpis['latest_hotfix_adoption_pct'] is not None else "N/A")
  kc5.metric("✅ Fleet Compliance",
             f"{_kpis['fleet_compliance_pct']}%" if _kpis['fleet_compliance_pct'] is not None else "N/A")

  st.markdown("---")

  # ── SECTION PICKER ────────────────────────────────────────────────
  fleet_sections = [
      "📊 Deployment & Upgrade Analysis",
      "🔍 Vessel Status Analysis",
      "⏱ Delay Analysis",
      "🏷 Common Status Analysis",
      "🔎 IMO-Based Lookup",
      "📈 KPI Metrics",
      "📉 Visualizations",
  ]
  fleet_section = st.selectbox("Jump to section", fleet_sections, key="fleet_section_2")
  st.markdown("---")

  def _fl(fig, h=420):
      fig.update_layout(margin=dict(t=45,b=30,l=25,r=25), height=h,
          font=dict(family="DM Sans, sans-serif", size=12),
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
      return fig

  # ════════════════════════════════════════════════════════════════
  # 1. DEPLOYMENT & UPGRADE ANALYSIS
  # ════════════════════════════════════════════════════════════════
  if "Deployment" in fleet_section:
      st.markdown("#### 📊 Deployment & Upgrade Analysis")

      if release_col and status_col:
          rel_grp = df.groupby(release_col)[status_col].apply(
              lambda x: x.astype(str).str.lower().str.contains("live|online|active|installed|success", na=False).sum()
          ).reset_index()
          rel_grp.columns = [release_col, "Successful"]
          rel_total = df.groupby(release_col)[status_col].count().reset_index()
          rel_total.columns = [release_col, "Total"]
          rel_df = rel_grp.merge(rel_total, on=release_col)
          rel_df["Success Rate %"] = (rel_df["Successful"] / rel_df["Total"] * 100).round(1)
          rel_df["Offline"] = rel_df["Total"] - rel_df["Successful"]

          da1, da2 = st.columns(2)
          with da1:
              st.markdown("**Q1 — Vessels completing each deployment**")
              st.dataframe(rel_df[[release_col, "Successful", "Total"]], use_container_width=True, hide_index=True)

              # Q2 latest hotfix %
              if len(rel_df) > 0:
                  latest_row = rel_df.iloc[-1]
                  st.metric(f"Q2 — Latest hotfix ({latest_row[release_col]}) adoption",
                            f"{latest_row['Success Rate %']}%")

          with da2:
              # Q3 highest success rate
              best = rel_df.loc[rel_df["Success Rate %"].idxmax()]
              st.metric("Q3 — Highest success rate hotfix",
                        f"{best[release_col]}", f"{best['Success Rate %']}%")
              # Q4 most offline
              worst = rel_df.loc[rel_df["Offline"].idxmax()]
              st.metric("Q4 — Most offline vessels", f"{worst[release_col]}", f"{int(worst['Offline'])} offline",
                        delta_color="inverse")

          # Q5 — Trend chart
          st.markdown("**Q5 — Deployment success trend across releases**")
          fig = go.Figure()
          fig.add_bar(x=rel_df[release_col], y=rel_df["Successful"],
                      name="Successful", marker_color="#22c55e")
          fig.add_bar(x=rel_df[release_col], y=rel_df["Offline"],
                      name="Offline", marker_color="#ef4444")
          fig.add_scatter(x=rel_df[release_col], y=rel_df["Success Rate %"],
                          mode="lines+markers", name="Success Rate %",
                          yaxis="y2", line=dict(color="#3b82f6", width=2.5))
          fig.update_layout(
              barmode="stack", yaxis2=dict(overlaying="y", side="right", title="Success Rate %"),
              title="Deployment Success by Release", template=_TPL,
              xaxis_tickangle=-35,
          )
          st.plotly_chart(_fl(fig, h=420), use_container_width=True)
      else:
          st.info("Upload a dataset with Release/Hotfix and Status columns to see deployment analysis.")

  # ════════════════════════════════════════════════════════════════
  # 2. VESSEL STATUS ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Vessel Status" in fleet_section:
      st.markdown("#### 🔍 Vessel Status Analysis")

      if status_col and id_col:
          vs1, vs2 = st.columns(2)
          vessel_latest = df.groupby(id_col)[status_col].last().reset_index()
          vessel_latest.columns = [id_col, "Current Status"]
          status_counts = vessel_latest["Current Status"].value_counts().reset_index()
          status_counts.columns = ["Status", "Count"]

          with vs1:
              st.markdown("**Q1 — Live vs Offline vessels**")
              fig_pie = px.pie(status_counts, names="Status", values="Count",
                  color_discrete_sequence=_COLORS, hole=0.4,
                  title="Current Vessel Status Distribution", template=_TPL)
              st.plotly_chart(_fl(fig_pie, h=380), use_container_width=True)

          with vs2:
              st.markdown("**Q3 — Fleet up-to-date %**")
              if release_col:
                  all_releases = df[release_col].dropna().astype(str).unique().tolist()
                  latest_rel = sorted(all_releases)[-1] if all_releases else None
                  if latest_rel:
                      latest_sub = df[df[release_col].astype(str) == latest_rel]
                      updated_ids = latest_sub[
                          latest_sub[status_col].astype(str).str.lower().str.contains(
                              "live|online|active|installed|success", na=False)
                      ][id_col].unique()
                      upd_pct = round(len(updated_ids) / _kpis["total_vessels"] * 100, 1) if _kpis["total_vessels"] else 0
                      st.metric(f"Vessels on latest release ({latest_rel})", f"{len(updated_ids)}", f"{upd_pct}% of fleet")

              st.markdown("**All vessel current statuses**")
              st.dataframe(vessel_latest.sort_values("Current Status"), use_container_width=True,
                           hide_index=True, height=280)

          # Q2 vessels offline for multiple releases
          if release_col:
              st.markdown("**Q2 — Vessels offline across multiple releases**")
              offline_mask = df[status_col].astype(str).str.lower().str.contains(
                  "offline|off|inactive|failed|missing|no", na=False)
              offline_count = df[offline_mask].groupby(id_col)[release_col].nunique().reset_index()
              offline_count.columns = [id_col, "Releases Offline"]
              offline_count = offline_count[offline_count["Releases Offline"] > 1].sort_values(
                  "Releases Offline", ascending=False)
              if len(offline_count):
                  fig_off = px.bar(offline_count.head(20), x=id_col, y="Releases Offline",
                      color="Releases Offline", color_continuous_scale="Reds",
                      title="Vessels Offline Across Multiple Releases", template=_TPL)
                  fig_off.update_xaxes(tickangle=-45)
                  st.plotly_chart(_fl(fig_off, h=380), use_container_width=True)
              else:
                  st.success("No vessels found offline across multiple releases.")

          # Q4 missed most updates
          if release_col:
              st.markdown("**Q4 — Vessels that missed the most updates**")
              missed = df[~df[status_col].astype(str).str.lower().str.contains(
                  "live|online|active|installed|success", na=False)
              ].groupby(id_col)[release_col].nunique().reset_index()
              missed.columns = [id_col, "Updates Missed"]
              missed = missed.sort_values("Updates Missed", ascending=False).head(15)
              st.dataframe(missed, use_container_width=True, hide_index=True)
      else:
          st.info("Upload a dataset with Vessel/IMO and Status columns for status analysis.")

  # ════════════════════════════════════════════════════════════════
  # 3. DELAY ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Delay" in fleet_section:
      st.markdown("#### ⏱ Delay Analysis")

      if delay_col and delay_col in df.columns and pd.api.types.is_numeric_dtype(df[delay_col]):
          dl1, dl2, dl3, dl4 = st.columns(4)
          avg_delay = df[delay_col].mean()
          dl1.metric("Q1 — Avg Deployment Delay", f"{avg_delay:.1f} days")
          on_time = int((df[delay_col] <= 0).sum())
          dl4.metric("Q4 — Installed On Time", f"{on_time:,}")

          # Q2 longest delay vessels
          if id_col:
              st.markdown("**Q2 — Vessels with longest install delays**")
              delay_by_vessel = df.groupby(id_col)[delay_col].max().reset_index()
              delay_by_vessel.columns = [id_col, "Max Delay (days)"]
              delay_by_vessel = delay_by_vessel.sort_values("Max Delay (days)", ascending=False).head(10)
              fig_dl = px.bar(delay_by_vessel, x=id_col, y="Max Delay (days)",
                  color="Max Delay (days)", color_continuous_scale="Oranges",
                  title="Top 10 Vessels by Maximum Deployment Delay", template=_TPL, text_auto=True)
              fig_dl.update_xaxes(tickangle=-45)
              st.plotly_chart(_fl(fig_dl, h=400), use_container_width=True)

          # Q3 releases with highest avg delay
          if release_col:
              st.markdown("**Q3 — Releases with highest average delay**")
              rel_delay = df.groupby(release_col)[delay_col].mean().round(1).reset_index()
              rel_delay.columns = [release_col, "Avg Delay (days)"]
              rel_delay = rel_delay.sort_values("Avg Delay (days)", ascending=False)
              fig_rd = px.bar(rel_delay, x=release_col, y="Avg Delay (days)",
                  color="Avg Delay (days)", color_continuous_scale="Reds",
                  title="Average Delay by Release", template=_TPL, text_auto=True)
              st.plotly_chart(_fl(fig_rd, h=380), use_container_width=True)

          # Histogram
          st.markdown("**Delay distribution across all deployments**")
          fig_hist = px.histogram(df, x=delay_col, nbins=40,
              title="Deployment Delay Distribution", template=_TPL,
              color_discrete_sequence=["#3b82f6"], marginal="rug")
          fig_hist.add_vline(x=avg_delay, line_dash="dash", line_color="red",
              annotation_text=f"Mean: {avg_delay:.1f}d")
          st.plotly_chart(_fl(fig_hist, h=380), use_container_width=True)

          # Q5 consistently late vessels
          if id_col:
              st.markdown("**Q5 — Vessels that consistently deploy late**")
              late_mask = df[delay_col] > 0
              late_pct = df[late_mask].groupby(id_col).size() / df.groupby(id_col).size()
              late_pct = late_pct.dropna().sort_values(ascending=False).head(15).reset_index()
              late_pct.columns = [id_col, "Late Deployment Rate"]
              late_pct["Late Deployment Rate"] = (late_pct["Late Deployment Rate"] * 100).round(1)
              st.dataframe(late_pct, use_container_width=True, hide_index=True)
      else:
          st.info("No numeric delay column detected. Make sure your dataset has a days/delay column.")

  # ════════════════════════════════════════════════════════════════
  # 4. COMMON STATUS ANALYSIS
  # ════════════════════════════════════════════════════════════════
  elif "Common Status" in fleet_section:
      st.markdown("#### 🏷 Common Status Analysis")

      cs_col = common_status_col or status_col
      if cs_col and cs_col in df.columns:
          cs1, cs2 = st.columns(2)
          cs_counts = df[cs_col].fillna("(No Status)").astype(str).value_counts().reset_index()
          cs_counts.columns = ["Status", "Count"]

          with cs1:
              st.markdown("**Q1/Q5 — Distribution of common statuses**")
              fig_cs = px.pie(cs_counts, names="Status", values="Count",
                  color_discrete_sequence=_COLORS, hole=0.45,
                  title="Common Status Distribution", template=_TPL)
              st.plotly_chart(_fl(fig_cs, h=380), use_container_width=True)

          with cs2:
              st.markdown("**Q3 — Vessels per status category**")
              if id_col:
                  vc_per = df.groupby(cs_col)[id_col].nunique().reset_index()
                  vc_per.columns = ["Status", "Vessel Count"]
                  vc_per = vc_per.sort_values("Vessel Count", ascending=False)
                  st.dataframe(vc_per, use_container_width=True, hide_index=True)

          # Q2 vessels sharing status
          st.markdown("**Q2 — Vessels sharing each status**")
          if id_col:
              shared = df[[id_col, cs_col]].drop_duplicates()
              for status_val, grp in shared.groupby(cs_col):
                  with st.expander(f"**{status_val}** — {len(grp)} vessel(s)"):
                      st.write(", ".join(grp[id_col].astype(str).tolist()))

          # Q4 vessels with no status
          no_status = df[df[cs_col].isna() | (df[cs_col].astype(str).str.strip() == "")]
          if len(no_status) > 0 and id_col:
              st.warning(f"**Q4 — {no_status[id_col].nunique()} vessels with no common status assigned**")
              st.dataframe(no_status[[id_col]].drop_duplicates(), use_container_width=True, hide_index=True)
          else:
              st.success("Q4 — All vessels have a common status assigned.")
      else:
          st.info("No common status column detected in the dataset.")

  # ════════════════════════════════════════════════════════════════
  # 5. IMO-BASED LOOKUP
  # ════════════════════════════════════════════════════════════════
  elif "IMO" in fleet_section:
      st.markdown("#### 🔎 IMO / Vessel Lookup")

      lookup_col = id_col
      if lookup_col:
          all_ids = sorted(df[lookup_col].dropna().astype(str).unique().tolist())

          imo_search = st.text_input("Search vessel by name / IMO", placeholder="Type to filter…", key="imo_search")
          filtered_ids = [v for v in all_ids if imo_search.lower() in v.lower()] if imo_search else all_ids
          selected_id = st.selectbox("Select vessel", filtered_ids, key="imo_select") if filtered_ids else None

          if selected_id:
              vessel_df = df[df[lookup_col].astype(str) == selected_id]
              st.markdown(f"### 🚢 {selected_id}")

              # KPI row for the vessel
              iv1, iv2, iv3, iv4 = st.columns(4)
              iv1.metric("Records", len(vessel_df))
              if status_col:
                  statuses = vessel_df[status_col].astype(str).unique()
                  iv2.metric("Unique Statuses", len(statuses))
              if release_col:
                  iv3.metric("Releases in data", vessel_df[release_col].nunique())
              if delay_col and pd.api.types.is_numeric_dtype(df[delay_col]):
                  iv4.metric("Avg Delay", f"{vessel_df[delay_col].mean():.1f} days")

              st.markdown("**Complete update history**")
              st.dataframe(vessel_df, use_container_width=True, hide_index=True)

          # Q3 compare two vessels
          st.markdown("---")
          st.markdown("**Compare two vessels**")
          cmp1, cmp2 = st.columns(2)
          with cmp1:
              v1 = st.selectbox("Vessel A", all_ids, key="cmp_v1")
          with cmp2:
              v2 = st.selectbox("Vessel B", all_ids, index=min(1, len(all_ids)-1), key="cmp_v2")
          if v1 and v2 and release_col and status_col:
              cmp_df = df[df[lookup_col].astype(str).isin([v1, v2])][[lookup_col, release_col, status_col]]
              pivot = cmp_df.pivot_table(index=release_col, columns=lookup_col,
                  values=status_col, aggfunc="last").reset_index()
              st.dataframe(pivot, use_container_width=True, hide_index=True)

          # Q4 vessels with missing update records
          if release_col:
              st.markdown("**Q4 — Vessels with missing update records**")
              all_rel = df[release_col].dropna().unique()
              expected = len(all_rel)
              actual = df.groupby(lookup_col)[release_col].nunique()
              missing_recs = actual[actual < expected].reset_index()
              missing_recs.columns = [lookup_col, "Releases Recorded"]
              missing_recs["Missing Releases"] = expected - missing_recs["Releases Recorded"]
              missing_recs = missing_recs.sort_values("Missing Releases", ascending=False)
              if len(missing_recs):
                  st.dataframe(missing_recs, use_container_width=True, hide_index=True)
              else:
                  st.success("All vessels have records for every release.")
      else:
          st.info("No IMO / vessel identifier column detected.")

  # ════════════════════════════════════════════════════════════════
  # 6. KPI METRICS
  # ════════════════════════════════════════════════════════════════
  elif "KPI" in fleet_section:
      st.markdown("#### 📈 KPI Metrics")

      kp1, kp2, kp3 = st.columns(3)
      kp4, kp5     = st.columns(2)

      # Fleet Compliance Rate
      kp1.metric("✅ Fleet Compliance Rate",
          f"{_kpis['fleet_compliance_pct']}%" if _kpis['fleet_compliance_pct'] is not None else "N/A")

      # Update Success Rate
      if status_col:
          total_rows = len(df)
          success_rows = df[status_col].astype(str).str.lower().str.contains(
              "live|online|active|installed|success", na=False).sum()
          update_success = round(success_rows / total_rows * 100, 1) if total_rows else 0
          kp2.metric("📦 Update Success Rate", f"{update_success}%")
      else:
          kp2.metric("📦 Update Success Rate", "N/A")

      # Offline Rate
      off_rate = round(_kpis["offline_vessels"] / _kpis["total_vessels"] * 100, 1) if _kpis["total_vessels"] else 0
      kp3.metric("🔴 Offline Rate", f"{off_rate}%", delta_color="inverse")

      # Avg Deployment Delay
      if delay_col and pd.api.types.is_numeric_dtype(df[delay_col]):
          kp4.metric("⏱ Avg Deployment Delay", f"{df[delay_col].mean():.1f} days")
      else:
          kp4.metric("⏱ Avg Deployment Delay", "N/A")

      # Latest Hotfix Adoption
      kp5.metric("🆕 Latest Hotfix Adoption",
          f"{_kpis['latest_hotfix_adoption_pct']}%" if _kpis['latest_hotfix_adoption_pct'] is not None else "N/A")

      # Summary table
      st.markdown("---")
      st.markdown("**Full KPI Summary Table**")
      kpi_rows = [
          {"KPI": "Total Vessels",            "Value": _kpis["total_vessels"]},
          {"KPI": "Live Vessels",             "Value": _kpis["live_vessels"]},
          {"KPI": "Offline Vessels",          "Value": _kpis["offline_vessels"]},
          {"KPI": "Fleet Compliance Rate (%)", "Value": _kpis["fleet_compliance_pct"]},
          {"KPI": "Latest Hotfix Adoption (%)", "Value": _kpis["latest_hotfix_adoption_pct"]},
          {"KPI": "Offline Rate (%)",          "Value": off_rate},
      ]
      if status_col:
          kpi_rows.append({"KPI": "Update Success Rate (%)", "Value": update_success})
      if delay_col and pd.api.types.is_numeric_dtype(df[delay_col]):
          kpi_rows.append({"KPI": "Avg Deployment Delay (days)", "Value": round(df[delay_col].mean(), 1)})
      st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

  # ════════════════════════════════════════════════════════════════
  # 7. VISUALIZATIONS
  # ════════════════════════════════════════════════════════════════
  elif "Visualiz" in fleet_section:
      st.markdown("#### 📉 Fleet Visualizations")

      viz_choice = st.selectbox("Choose chart", [
          "Deployment Success by Hotfix (Bar)",
          "Live vs Offline Vessels (Pie)",
          "Update Timeline (Line)",
          "Delay Distribution (Histogram)",
          "Compliance Status (Donut)",
          "Top 10 Delayed Vessels (Bar)",
          "Vessel × Hotfix Heatmap",
      ], key="fleet_viz")

      if viz_choice == "Deployment Success by Hotfix (Bar)":
          if release_col and status_col:
              rel_grp = df.groupby(release_col)[status_col].apply(
                  lambda x: x.astype(str).str.lower().str.contains("live|online|active|installed|success", na=False).sum()
              ).reset_index()
              rel_grp.columns = [release_col, "Successful"]
              rel_total = df.groupby(release_col)[status_col].count().reset_index()
              rel_total.columns = [release_col, "Total"]
              rel_df2 = rel_grp.merge(rel_total, on=release_col)
              rel_df2["Failed"] = rel_df2["Total"] - rel_df2["Successful"]
              fig = go.Figure()
              fig.add_bar(x=rel_df2[release_col], y=rel_df2["Successful"], name="Success", marker_color="#22c55e")
              fig.add_bar(x=rel_df2[release_col], y=rel_df2["Failed"], name="Failed/Offline", marker_color="#ef4444")
              fig.update_layout(barmode="group", title="Deployment Success by Hotfix",
                  template=_TPL, xaxis_tickangle=-35)
              st.plotly_chart(_fl(fig, h=430), use_container_width=True)
          else:
              st.info("Needs Release and Status columns.")

      elif viz_choice == "Live vs Offline Vessels (Pie)":
          if status_col and id_col:
              vessel_latest2 = df.groupby(id_col)[status_col].last().reset_index()
              vessel_latest2.columns = [id_col, "Status"]
              fig = px.pie(vessel_latest2["Status"].value_counts().reset_index(),
                  names="Status", values="count", hole=0.35,
                  color_discrete_sequence=_COLORS, title="Live vs Offline Vessels", template=_TPL)
              st.plotly_chart(_fl(fig, h=420), use_container_width=True)
          else:
              st.info("Needs Status and Vessel/IMO columns.")

      elif viz_choice == "Update Timeline (Line)":
          if date_col and status_col:
              try:
                  tl_df = df.copy()
                  tl_df[date_col] = pd.to_datetime(tl_df[date_col], errors="coerce")
                  tl_df = tl_df.dropna(subset=[date_col])
                  tl_df["_success"] = tl_df[status_col].astype(str).str.lower().str.contains(
                      "live|online|active|installed|success", na=False).astype(int)
                  tl_grp = tl_df.groupby(tl_df[date_col].dt.to_period("M").astype(str))["_success"].mean().reset_index()
                  tl_grp.columns = ["Month", "Success Rate"]
                  tl_grp["Success Rate"] = (tl_grp["Success Rate"] * 100).round(1)
                  fig = px.line(tl_grp, x="Month", y="Success Rate",
                      title="Monthly Deployment Success Rate", template=_TPL,
                      color_discrete_sequence=["#3b82f6"], markers=True)
                  fig.update_traces(line_width=2.5)
                  fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% target")
                  st.plotly_chart(_fl(fig, h=420), use_container_width=True)
              except Exception as e:
                  st.error(f"Timeline error: {e}")
          else:
              st.info("Needs a Date column and Status column.")

      elif viz_choice == "Delay Distribution (Histogram)":
          if delay_col and pd.api.types.is_numeric_dtype(df[delay_col]):
              fig = px.histogram(df, x=delay_col, nbins=40, marginal="box",
                  title="Deployment Delay Distribution", template=_TPL,
                  color_discrete_sequence=["#3b82f6"])
              fig.add_vline(x=df[delay_col].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {df[delay_col].mean():.1f}d")
              st.plotly_chart(_fl(fig, h=430), use_container_width=True)
          else:
              st.info("Needs a numeric Delay/Days column.")

      elif viz_choice == "Compliance Status (Donut)":
          if _kpis["fleet_compliance_pct"] is not None:
              comp_val = _kpis["fleet_compliance_pct"]
              non_comp = 100 - comp_val
              fig = go.Figure(go.Pie(
                  values=[comp_val, non_comp],
                  labels=["Compliant", "Non-Compliant"],
                  hole=0.6,
                  marker_colors=["#22c55e", "#ef4444"],
              ))
              fig.update_layout(
                  title="Fleet Compliance Status", template=_TPL,
                  annotations=[dict(text=f"{comp_val}%", font_size=22, showarrow=False)]
              )
              st.plotly_chart(_fl(fig, h=400), use_container_width=True)
          else:
              st.info("Needs Status and Vessel columns.")

      elif viz_choice == "Top 10 Delayed Vessels (Bar)":
          if delay_col and pd.api.types.is_numeric_dtype(df[delay_col]) and id_col:
              top_delay = df.groupby(id_col)[delay_col].mean().round(1).sort_values(ascending=False).head(10).reset_index()
              top_delay.columns = [id_col, "Avg Delay (days)"]
              fig = px.bar(top_delay, y=id_col, x="Avg Delay (days)", orientation="h",
                  color="Avg Delay (days)", color_continuous_scale="Reds",
                  title="Top 10 Vessels by Average Deployment Delay", template=_TPL, text_auto=True)
              fig.update_yaxes(categoryorder="total ascending")
              st.plotly_chart(_fl(fig, h=430), use_container_width=True)
          else:
              st.info("Needs Delay and Vessel/IMO columns.")

      elif viz_choice == "Vessel × Hotfix Heatmap":
          if release_col and status_col and id_col:
              hm_df = df[[id_col, release_col, status_col]].copy()
              hm_df["_val"] = hm_df[status_col].astype(str).str.lower().str.contains(
                  "live|online|active|installed|success", na=False).astype(int)
              # Limit to top 30 vessels and all releases for readability
              top_vessels = hm_df.groupby(id_col)["_val"].mean().nsmallest(30).index.tolist()
              hm_sub = hm_df[hm_df[id_col].isin(top_vessels)]
              pivot_hm = hm_sub.pivot_table(index=id_col, columns=release_col,
                  values="_val", aggfunc="mean").fillna(0)
              fig = go.Figure(go.Heatmap(
                  z=pivot_hm.values,
                  x=pivot_hm.columns.tolist(),
                  y=pivot_hm.index.tolist(),
                  colorscale=[[0, "#ef4444"], [1, "#22c55e"]],
                  text=pivot_hm.values.round(0).astype(int),
                  texttemplate="%{text}",
                  textfont={"size": 10},
                  colorbar=dict(title="", tickvals=[0, 1], ticktext=["Offline", "Live"]),
              ))
              fig.update_layout(
                  title="Vessel × Hotfix Deployment Heatmap (30 lowest-compliance vessels)",
                  template=_TPL,
                  height=max(400, len(pivot_hm) * 22),
                  margin=dict(t=55, b=30, l=140, r=20),
                  xaxis_tickangle=-35,
                  yaxis=dict(autorange="reversed"),
              )
              st.plotly_chart(fig, use_container_width=True)
          else:
              st.info("Needs Release, Status, and Vessel/IMO columns.")


# ── Data Preview Page ─────────────────────────
if _active == "🗃 Data Preview" and df is not None:
      # ── Multi-file source summary (only shown when >1 file merged) ──
      if "_source_file" in df.columns:
          src_counts = df["_source_file"].value_counts().reset_index()
          src_counts.columns = ["File", "Rows"]
          src_counts["% of Total"] = (src_counts["Rows"] / len(df) * 100).round(1).astype(str) + "%"
          with st.expander(f"📂 {src_counts.shape[0]} files merged — click to see breakdown", expanded=True):
              st.dataframe(src_counts, use_container_width=True, hide_index=True)
              _filter_file = st.selectbox(
                  "Filter to one file (optional)",
                  ["All files"] + src_counts["File"].tolist(),
                  key="dp_file_filter",
              )
              if _filter_file != "All files":
                  df = df[df["_source_file"] == _filter_file]
                  st.caption(f"Showing {len(df):,} rows from **{_filter_file}**")
          st.markdown("---")

      st.markdown(f"**Showing first 200 rows of {len(df):,} total**")

      col_filter, col_search = st.columns([3, 1])
      with col_filter:
          selected_cols = st.multiselect(
              "Show columns",
              df.columns.tolist(),
              default=df.columns.tolist()[:10]
          )
      with col_search:
          search_term = st.text_input("Search value", placeholder="Filter rows…")

      display_df = df[selected_cols] if selected_cols else df
      if search_term:
          mask = display_df.astype(str).apply(
              lambda col: col.str.contains(search_term, case=False, na=False)
          ).any(axis=1)
          display_df = display_df[mask]
          st.caption(f"Found {len(display_df):,} matching rows")

      st.dataframe(
          display_df.head(200),
          use_container_width=True,
          height=450
      )

      # Download button
      csv_buf = io.StringIO()
      df.to_csv(csv_buf, index=False)
      st.download_button(
          "⬇ Download full dataset as CSV",
          csv_buf.getvalue(),
          file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
          mime="text/csv"
      )


# ── Column Profile Page ────────────────────────
if _active == "🔍 Column Profile" and df is not None:
      profile = st.session_state.profile
      st.markdown(f"**{len(df.columns)} columns profiled**")

      # Summary row
      m1, m2, m3, m4 = st.columns(4)
      total_missing = sum(p["missing"] for p in profile.values())
      numeric_cols = sum(1 for p in profile.values() if p["is_numeric"])
      avg_fill = round(sum(p["fill_pct"] for p in profile.values()) / len(profile), 1)

      m1.metric("Total Columns", len(df.columns))
      m2.metric("Numeric Columns", numeric_cols)
      m3.metric("Avg Fill Rate", f"{avg_fill}%")
      m4.metric("Total Missing", f"{total_missing:,}")

      st.markdown("---")

      # Per-column cards
      for col in df.columns:
          p = profile[col]
          fill_color = "#22c55e" if p["fill_pct"] >= 90 else "#f59e0b" if p["fill_pct"] >= 60 else "#ef4444"

          with st.expander(f"{'🔢' if p['is_numeric'] else '🔤'} **{col}** — {p['fill_pct']}% filled · {p['unique']} unique", expanded=False):
              c1, c2, c3 = st.columns(3)
              c1.metric("Fill Rate", f"{p['fill_pct']}%", delta=f"-{p['missing_pct']}% missing" if p["missing_pct"] > 0 else "No missing")
              c2.metric("Unique Values", f"{p['unique']:,}")
              c3.metric("Missing", f"{p['missing']:,}")

              if p["is_numeric"]:
                  n1, n2, n3, n4, n5 = st.columns(5)
                  fmt2 = lambda v: f"{v:,.2f}" if v is not None else "—"
                  fmt4 = lambda v: f"{v:,.4f}" if v is not None else "—"
                  n1.metric("Min", fmt2(p['min']))
                  n2.metric("Max", fmt2(p['max']))
                  n3.metric("Mean", fmt4(p['mean']))
                  n4.metric("Median", fmt4(p['median']))
                  n5.metric("Outliers (IQR)", f"{p['outliers']}" if p['outliers'] is not None else "—")

                  # Mini histogram
                  non_null = df[col].dropna()
                  if len(non_null) > 0:
                      hist_counts = pd.cut(non_null, bins=20).value_counts().sort_index()
                      hist_df = pd.DataFrame({
                          "range": [str(round(i.left, 2)) for i in hist_counts.index],
                          "count": hist_counts.values
                      })
                      st.bar_chart(hist_df.set_index("range"), height=120, use_container_width=True)
              else:
                  st.markdown("**Top Values:**")
                  top_df = pd.DataFrame(
                      list(p["top_values"].items()),
                      columns=["Value", "Count"]
                  )
                  top_df["% of Total"] = (top_df["Count"] / len(df) * 100).round(1).astype(str) + "%"
                  st.dataframe(top_df, use_container_width=True, hide_index=True, height=min(200, 40 + len(top_df) * 35))


# ── Statistics Page ────────────────────────────
if _active == "📈 Statistics" and df is not None:
      st.markdown("**Descriptive Statistics**")

      num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
      cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

      if num_cols:
          st.markdown("##### Numeric Columns")
          desc = df[num_cols].describe().T
          desc = desc.round(4)
          st.dataframe(desc, use_container_width=True)

      if cat_cols:
          st.markdown("##### Categorical Columns (top 5 values each)")
          cat_summary = []
          for col in cat_cols[:10]:
              vc = df[col].fillna("(blank)").astype(str).value_counts().head(5)
              for val, cnt in vc.items():
                  cat_summary.append({
                      "Column": col,
                      "Value": val,
                      "Count": cnt,
                      "% of Total": f"{cnt / len(df) * 100:.1f}%"
                  })
          if cat_summary:
              st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True)

      # Correlation heatmap (numeric only)
      if len(num_cols) >= 2:
          st.markdown("##### Correlation Matrix (numeric columns)")
          corr = df[num_cols].corr().round(3)
          st.dataframe(
              corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
              use_container_width=True
          )



# ── Visual Analytics Page (Phase 1: AI Auto-Charts + 1 manual customization) ──
if _active == "📊 Visual Analytics" and df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Auto-detect date columns
    date_cols = []
    for _c in all_cols:
        if "datetime" in str(df[_c].dtype):
            date_cols.append(_c)
        elif df[_c].dtype == object:
            try:
                _s = pd.to_datetime(df[_c].dropna().head(50), errors="coerce")
                if _s.notna().sum() > 30:
                    date_cols.append(_c)
            except Exception:
                pass

    theme = get_theme()
    TPL    = theme["template"]
    COLORS = theme["colors"]
    PALETTES = ["Blues","Greens","Reds","Purples","Oranges","Teal","Magenta"]

    # ── KPI Banner ──────────────────────────────────────────────────────
    st.markdown("### 📊 Visual Analytics")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", len(df.columns))
    k3.metric("Numeric", len(num_cols))
    k4.metric("Categorical", len(cat_cols))
    k5.metric("Date cols", len(date_cols))
    missing_pct = round(df.isna().sum().sum()/(len(df)*len(df.columns))*100,1) if len(df)>0 else 0
    k6.metric("Missing %", f"{missing_pct}%")
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — AI AUTO-GENERATED CHARTS
    # ═══════════════════════════════════════════════════════════════
    st.markdown("#### 🤖 AI-Suggested Charts")
    st.caption("Automatically generated based on your dataset. Charts refresh when you upload a new file or click Regenerate.")

    # Build a stable cache key from file + columns + provider + model
    _col_sig = "|".join(all_cols[:20])
    _cache_key = f"{st.session_state.file_name}|{_col_sig}|{st.session_state.provider}|{st.session_state.model}"

    # Check if we have cached suggestions for this file+provider
    if "ai_charts_cache_key" not in st.session_state or st.session_state.ai_charts_cache_key != _cache_key:
        st.session_state.ai_charts_suggestions = None
        st.session_state.ai_charts_cache_key = _cache_key

    regen_col, info_col = st.columns([1, 5])
    with regen_col:
        if st.button("🔄 Regenerate Charts", key="regen_ai_charts", use_container_width=True):
            st.session_state.ai_charts_suggestions = None
            st.session_state.ai_charts_cache_key = ""

    with info_col:
        if not st.session_state.api_key:
            st.warning("⚠️ Add an API key in the sidebar to enable AI chart suggestions.")

    # Load suggestions if not cached
    if st.session_state.get("ai_charts_suggestions") is None and st.session_state.api_key:
        with st.spinner("🤖 AI is analyzing your data and selecting the best charts…"):
            suggestions = ai_suggest_charts(df, st.session_state.profile)
            st.session_state.ai_charts_suggestions = suggestions
            st.session_state.ai_charts_cache_key = _cache_key
    elif st.session_state.get("ai_charts_suggestions") is None:
        # No API key — generate some sensible defaults without AI
        suggestions = []
        if cat_cols and num_cols:
            suggestions.append({"type":"bar","x":cat_cols[0],"y":num_cols[0],"title":f"{num_cols[0]} by {cat_cols[0]}","insight":f"Distribution of {num_cols[0]} across {cat_cols[0]} categories."})
        if len(num_cols) >= 2:
            suggestions.append({"type":"scatter","x":num_cols[0],"y":num_cols[1],"title":f"{num_cols[0]} vs {num_cols[1]}","insight":f"Correlation between {num_cols[0]} and {num_cols[1]}."})
        if len(num_cols) >= 2:
            suggestions.append({"type":"heatmap","x":None,"y":None,"title":"Correlation Heatmap","insight":"Correlations between all numeric columns."})
        if cat_cols:
            suggestions.append({"type":"pie","x":cat_cols[0],"y":None,"title":f"Distribution of {cat_cols[0]}","insight":f"Proportional breakdown of {cat_cols[0]}."})
        if num_cols:
            suggestions.append({"type":"histogram","x":num_cols[0],"y":None,"title":f"Distribution of {num_cols[0]}","insight":f"Frequency distribution of {num_cols[0]}."})
        st.session_state.ai_charts_suggestions = suggestions
    else:
        suggestions = st.session_state.ai_charts_suggestions

    # Render AI charts in a 2-column grid
    if suggestions:
        for row_start in range(0, len(suggestions), 2):
            cols_pair = st.columns(2)
            for col_idx, cfg in enumerate(suggestions[row_start:row_start+2]):
                with cols_pair[col_idx]:
                    st.markdown(f"**{cfg.get('title', 'Chart')}**")
                    render_ai_chart(cfg, df, chart_idx=row_start + col_idx)
    else:
        st.info("No chart suggestions available. Check your API key in the sidebar.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # MANUAL CHART BUILDER (kept as expander — all original sections)
    # ═══════════════════════════════════════════════════════════════
    with st.expander("🛠 Advanced / Manual Chart Builder", expanded=False):
        st.caption("Full control over all chart types and settings. All original sections available below.")

        sections = [
            "1️⃣ Distribution",
            "2️⃣ Comparison",
            "3️⃣ Relationship / Scatter",
            "4️⃣ Composition",
            "5️⃣ Time Series",
            "6️⃣ Ranking",
            "7️⃣ Correlation",
            "8️⃣ Multi-Group Comparison",
            "9️⃣ Custom Chart Builder",
            "🔟 Missing Data Map",
            "1️⃣1️⃣ Pair Plot",
        ]
        active_section = st.selectbox("Jump to section", sections, key="viz_section")
        st.markdown("---")

        def std_layout(fig, h=420):
            fig.update_layout(
                margin=dict(t=45,b=30,l=25,r=25), height=h,
                font=dict(family="DM Sans, sans-serif", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template=TPL,
            )
            return fig

        # ═══════════════════════════════════════════════════════════════
        # 1️⃣ DISTRIBUTION
        # ═══════════════════════════════════════════════════════════════
        if "Distribution" in active_section:
            st.markdown("#### 1️⃣ Distribution Explorer")
            ctrl, chart_area = st.columns([1,3])
            with ctrl:
                dist_col  = st.selectbox("Column", all_cols, key="d_col")
                dist_type = st.selectbox("Chart type", ["Histogram","Box Plot","Violin","ECDF","Strip Plot","Bar (Top N)"], key="d_type")
                dist_pal  = st.selectbox("Color palette", PALETTES, key="d_pal")
                dist_bins = st.slider("Bins (histogram)", 5, 100, 30, key="d_bins")
                dist_topn = st.slider("Top N (bar)", 5, 50, 20, key="d_topn")
                dist_grp  = st.selectbox("Split/Color by", ["None"]+cat_cols, key="d_grp")
                show_mean = st.checkbox("Show mean line", True, key="d_mean")

            with chart_area:
                grp_arg = dist_grp if dist_grp != "None" else None
                if pd.api.types.is_numeric_dtype(df[dist_col]):
                    if dist_type == "Histogram":
                        fig = px.histogram(df, x=dist_col, nbins=dist_bins, color=grp_arg,
                            color_discrete_sequence=COLORS, template=TPL,
                            title=f"Distribution of {dist_col}", marginal="rug",
                            barmode="overlay", opacity=0.75)
                        if show_mean:
                            mean_val = df[dist_col].mean()
                            fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {mean_val:.2f}")
                    elif dist_type == "Box Plot":
                        fig = px.box(df, y=dist_col, x=grp_arg, color=grp_arg,
                            color_discrete_sequence=COLORS, template=TPL,
                            title=f"Box Plot — {dist_col}", points="outliers")
                    elif dist_type == "Violin":
                        fig = px.violin(df, y=dist_col, x=grp_arg, color=grp_arg,
                            color_discrete_sequence=COLORS, template=TPL,
                            title=f"Violin — {dist_col}", box=True, points="outliers")
                    elif dist_type == "ECDF":
                        fig = px.ecdf(df, x=dist_col, color=grp_arg,
                            color_discrete_sequence=COLORS, template=TPL,
                            title=f"ECDF — {dist_col}")
                    elif dist_type == "Strip Plot":
                        fig = px.strip(df, y=dist_col, x=grp_arg, color=grp_arg,
                            color_discrete_sequence=COLORS, template=TPL,
                            title=f"Strip Plot — {dist_col}")
                    else:
                        vc = df[dist_col].dropna().astype(str).value_counts().head(dist_topn).reset_index()
                        vc.columns=["Value","Count"]
                        fig = px.bar(vc, x="Value", y="Count", color="Count",
                            color_continuous_scale=dist_pal, template=TPL,
                            title=f"Top {dist_topn} values — {dist_col}")
                else:
                    vc = df[dist_col].fillna("(blank)").astype(str).value_counts().head(dist_topn).reset_index()
                    vc.columns=["Value","Count"]
                    if dist_type in ("Histogram","Bar (Top N)"):
                        fig = px.bar(vc, x="Value", y="Count", color="Count",
                            color_continuous_scale=dist_pal, template=TPL,
                            title=f"Top {dist_topn} — {dist_col}")
                        fig.update_xaxes(tickangle=-35)
                    elif dist_type == "Box Plot":
                        st.info("Box plot needs a numeric column. Showing bar instead.")
                        fig = px.bar(vc, x="Value", y="Count", template=TPL)
                    else:
                        fig = px.pie(vc.head(12), names="Value", values="Count",
                            title=f"Distribution — {dist_col}", template=TPL,
                            color_discrete_sequence=COLORS, hole=0.3)
                st.plotly_chart(std_layout(fig), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 2️⃣ COMPARISON
        # ═══════════════════════════════════════════════════════════════
        elif "Comparison" in active_section:
            st.markdown("#### 2️⃣ Comparison")
            c1,c2,c3 = st.columns(3)
            with c1:
                cmp_grp   = st.selectbox("Group by (X)", cat_cols if cat_cols else all_cols, key="cmp_grp")
                cmp_val   = st.selectbox("Value (Y)", ["Row Count"]+num_cols, key="cmp_val")
                cmp_agg   = st.selectbox("Aggregation", ["Count","Sum","Mean","Median","Max","Min","Std"], key="cmp_agg")
            with c2:
                cmp_chart = st.selectbox("Chart type", ["Vertical Bar","Horizontal Bar","Grouped Bar","Lollipop","Waterfall","Bullet"], key="cmp_chart")
                cmp_color = st.selectbox("Color by", ["None"]+cat_cols, key="cmp_color")
                cmp_topn  = st.slider("Top N", 5, 50, 20, key="cmp_topn")
            with c3:
                cmp_sort  = st.radio("Sort", ["Descending","Ascending","Alphabetical"], key="cmp_sort")
                cmp_pal   = st.selectbox("Palette", PALETTES, key="cmp_pal")
                show_vals = st.checkbox("Show values on bars", True, key="cmp_showval")

            agg_map = {"Count":"count","Sum":"sum","Mean":"mean","Median":"median","Max":"max","Min":"min","Std":"std"}
            if cmp_val == "Row Count":
                cmp_df = df[cmp_grp].fillna("(blank)").astype(str).value_counts().head(cmp_topn).reset_index()
                cmp_df.columns=[cmp_grp,"Value"]
                y_label = "Count"
            else:
                cmp_df = (df.groupby(df[cmp_grp].fillna("(blank)").astype(str))[cmp_val]
                    .agg(agg_map[cmp_agg]).round(2).reset_index())
                cmp_df.columns=[cmp_grp,"Value"]
                y_label = f"{cmp_agg} of {cmp_val}"

            if cmp_sort == "Descending":
                cmp_df = cmp_df.sort_values("Value", ascending=False).head(cmp_topn)
            elif cmp_sort == "Ascending":
                cmp_df = cmp_df.sort_values("Value", ascending=True).head(cmp_topn)
            else:
                cmp_df = cmp_df.sort_values(cmp_grp).head(cmp_topn)

            color_arg = cmp_color if cmp_color != "None" else None

            if cmp_chart == "Vertical Bar":
                fig = px.bar(cmp_df, x=cmp_grp, y="Value", color=cmp_grp if not color_arg else color_arg,
                    color_discrete_sequence=COLORS, color_continuous_scale=cmp_pal,
                    title=f"{y_label} by {cmp_grp}", template=TPL, text_auto=show_vals)
                fig.update_xaxes(tickangle=-35)
            elif cmp_chart == "Horizontal Bar":
                fig = px.bar(cmp_df, y=cmp_grp, x="Value", orientation="h",
                    color="Value", color_continuous_scale=cmp_pal,
                    title=f"{y_label} by {cmp_grp}", template=TPL, text_auto=show_vals)
            elif cmp_chart == "Grouped Bar":
                if color_arg and color_arg in df.columns:
                    sub = df.groupby([cmp_grp, color_arg])[cmp_val if cmp_val!="Row Count" else cmp_grp].agg(
                        agg_map[cmp_agg] if cmp_val!="Row Count" else "count").round(2).reset_index()
                    sub.columns = [cmp_grp, color_arg, "Value"]
                    fig = px.bar(sub, x=cmp_grp, y="Value", color=color_arg,
                        barmode="group", template=TPL, color_discrete_sequence=COLORS,
                        title=f"Grouped — {y_label} by {cmp_grp} & {color_arg}", text_auto=show_vals)
                    fig.update_xaxes(tickangle=-35)
                else:
                    st.warning("Select a 'Color by' column for grouped bar.")
                    fig = px.bar(cmp_df, x=cmp_grp, y="Value", template=TPL)
            elif cmp_chart == "Lollipop":
                fig = go.Figure()
                for _, row in cmp_df.iterrows():
                    fig.add_shape(type="line", x0=row[cmp_grp], x1=row[cmp_grp], y0=0, y1=row["Value"],
                        line=dict(color="#3b82f6", width=2))
                fig.add_trace(go.Scatter(x=cmp_df[cmp_grp], y=cmp_df["Value"],
                    mode="markers+text" if show_vals else "markers",
                    marker=dict(size=12, color="#3b82f6"),
                    text=cmp_df["Value"].round(1) if show_vals else None,
                    textposition="top center"))
                fig.update_layout(title=f"{y_label} by {cmp_grp}", template=TPL,
                    xaxis_tickangle=-35)
            elif cmp_chart == "Waterfall":
                fig = go.Figure(go.Waterfall(
                    name="value", orientation="v",
                    x=cmp_df[cmp_grp].tolist(),
                    y=cmp_df["Value"].tolist(),
                    connector={"line":{"color":"#94a3b8"}},
                ))
                fig.update_layout(title=f"Waterfall — {y_label}", template=TPL)
            else:  # Bullet
                fig = go.Figure()
                max_v = cmp_df["Value"].max()
                for i, row in cmp_df.iterrows():
                    fig.add_trace(go.Indicator(
                        mode="number+gauge", value=row["Value"],
                        domain={"x":[0,1],"y":[(i)/len(cmp_df),(i+0.85)/len(cmp_df)]},
                        title={"text": str(row[cmp_grp])[:20]},
                        gauge={"axis":{"range":[0,max_v]},
                               "bar":{"color":"#3b82f6"},
                               "bgcolor":"#e2e8f0"}
                    ))
                fig.update_layout(title=f"Bullet — {y_label}", height=max(400, len(cmp_df)*60))

            st.plotly_chart(std_layout(fig), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 3️⃣ RELATIONSHIP / SCATTER
        # ═══════════════════════════════════════════════════════════════
        elif "Relationship" in active_section:
            st.markdown("#### 3️⃣ Relationship / Scatter")
            if len(num_cols) < 2:
                st.info("Need at least 2 numeric columns.")
            else:
                r1,r2,r3,r4 = st.columns(4)
                with r1:
                    rx = st.selectbox("X axis", num_cols, index=0, key="rx")
                    ry = st.selectbox("Y axis", num_cols, index=min(1,len(num_cols)-1), key="ry")
                with r2:
                    rcolor = st.selectbox("Color by", ["None"]+cat_cols+num_cols, key="rcolor")
                    rsize  = st.selectbox("Size by", ["None"]+num_cols, key="rsize")
                with r3:
                    rstype = st.selectbox("Chart", ["Scatter","Bubble","Density Contour","Density Heatmap","Line"], key="rstype")
                    rtrendline = st.selectbox("Trendline", ["None","OLS","Lowess"], key="rtl")
                with r4:
                    rfacet = st.selectbox("Facet by (small multiples)", ["None"]+cat_cols, key="rfacet")
                    rsample = st.slider("Max points", 100, 5000, 2000, 100, key="rsample")

                plot_df = df.sample(min(rsample, len(df)))
                kw = dict(data_frame=plot_df, x=rx, y=ry, template=TPL, opacity=0.65,
                    title=f"{rx} vs {ry}")
                if rcolor != "None": kw["color"] = rcolor
                if rfacet != "None": kw["facet_col"] = rfacet; kw["facet_col_wrap"] = 3
                tl_map = {"OLS":"ols","Lowess":"lowess","None":None}
                tl = tl_map[rtrendline]

                if rstype == "Scatter":
                    if tl: kw["trendline"] = tl
                    fig = px.scatter(**kw, color_discrete_sequence=COLORS)
                elif rstype == "Bubble":
                    if rsize != "None":
                        kw["size"] = rsize; kw["size_max"] = 25
                    fig = px.scatter(**kw, color_discrete_sequence=COLORS)
                elif rstype == "Density Contour":
                    fig = px.density_contour(**kw, color_discrete_sequence=COLORS)
                    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
                elif rstype == "Density Heatmap":
                    fig = px.density_heatmap(plot_df, x=rx, y=ry, nbinsx=30, nbinsy=30,
                        color_continuous_scale="Blues", template=TPL, title=f"{rx} vs {ry}")
                else:  # Line
                    ldf = df[[rx,ry]].dropna().sort_values(rx)
                    fig = px.line(ldf, x=rx, y=ry, template=TPL, title=f"{ry} vs {rx}",
                        color_discrete_sequence=["#3b82f6"])

                st.plotly_chart(std_layout(fig, h=480), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 4️⃣ COMPOSITION
        # ═══════════════════════════════════════════════════════════════
        elif "Composition" in active_section:
            st.markdown("#### 4️⃣ Composition")
            p1,p2,p3 = st.columns(3)
            with p1:
                comp_grp  = st.selectbox("Primary group", cat_cols if cat_cols else all_cols, key="comp_grp")
                comp_val  = st.selectbox("Value", ["Row Count"]+num_cols, key="comp_val")
                comp_agg  = st.selectbox("Aggregation", ["Count","Sum","Mean"], key="comp_agg")
            with p2:
                comp_type = st.selectbox("Chart type", ["Pie","Donut","Sunburst","Treemap","Funnel","Stacked Bar","100% Stacked Bar"], key="comp_type")
                comp_sub  = st.selectbox("Sub-group (Sunburst/Treemap)", ["None"]+cat_cols, key="comp_sub")
            with p3:
                comp_topn = st.slider("Top N", 3, 30, 10, key="comp_topn")
                comp_pal  = st.selectbox("Palette", PALETTES, key="comp_pal")

            agg_map2 = {"Count":"count","Sum":"sum","Mean":"mean"}
            if comp_val == "Row Count":
                base = df[comp_grp].fillna("(blank)").astype(str).value_counts().head(comp_topn).reset_index()
                base.columns=[comp_grp,"Value"]
            else:
                base = (df.groupby(df[comp_grp].fillna("(blank)").astype(str))[comp_val]
                    .agg(agg_map2[comp_agg]).round(2).reset_index()
                    .sort_values(comp_val, ascending=False).head(comp_topn))
                base.columns=[comp_grp,"Value"]

            title4 = f"{comp_agg} of {comp_val} by {comp_grp}" if comp_val!="Row Count" else f"Count by {comp_grp}"

            if comp_type == "Pie":
                fig = px.pie(base, names=comp_grp, values="Value", title=title4, template=TPL,
                    color_discrete_sequence=COLORS)
            elif comp_type == "Donut":
                fig = px.pie(base, names=comp_grp, values="Value", title=title4, template=TPL,
                    hole=0.45, color_discrete_sequence=COLORS)
            elif comp_type == "Sunburst":
                if comp_sub != "None" and comp_sub in df.columns:
                    sun_df = df[[comp_grp, comp_sub]].fillna("(blank)").astype(str)
                    sun_df["Value"] = 1
                    if comp_val != "Row Count":
                        sun_df["Value"] = df[comp_val].fillna(0)
                    fig = px.sunburst(sun_df, path=[comp_grp, comp_sub], values="Value",
                        title=f"Sunburst: {comp_grp} → {comp_sub}", template=TPL,
                        color_discrete_sequence=COLORS)
                else:
                    fig = px.sunburst(base, path=[comp_grp], values="Value",
                        title=title4, template=TPL, color_discrete_sequence=COLORS)
            elif comp_type == "Treemap":
                if comp_sub != "None" and comp_sub in df.columns:
                    tm_df = df[[comp_grp, comp_sub]].fillna("(blank)").astype(str)
                    tm_df["Value"] = 1
                    if comp_val != "Row Count":
                        tm_df["Value"] = df[comp_val].fillna(0)
                    fig = px.treemap(tm_df, path=[comp_grp, comp_sub], values="Value",
                        title=f"Treemap: {comp_grp} → {comp_sub}",
                        color_continuous_scale=comp_pal)
                else:
                    fig = px.treemap(base, path=[comp_grp], values="Value",
                        title=title4, color_continuous_scale=comp_pal)
            elif comp_type == "Funnel":
                fig = px.funnel(base.sort_values("Value", ascending=False),
                    x="Value", y=comp_grp, title=title4, template=TPL,
                    color_discrete_sequence=COLORS)
            elif comp_type == "Stacked Bar":
                if comp_sub != "None" and comp_sub in df.columns:
                    sb_df = df.groupby([comp_grp, comp_sub]).size().reset_index(name="Value")
                    fig = px.bar(sb_df, x=comp_grp, y="Value", color=comp_sub,
                        barmode="stack", template=TPL, title=f"Stacked: {comp_grp} by {comp_sub}",
                        color_discrete_sequence=COLORS)
                    fig.update_xaxes(tickangle=-35)
                else:
                    fig = px.bar(base, x=comp_grp, y="Value", color=comp_grp,
                        template=TPL, title=title4, color_discrete_sequence=COLORS)
            else:  # 100% Stacked
                if comp_sub != "None" and comp_sub in df.columns:
                    sb_df = df.groupby([comp_grp, comp_sub]).size().reset_index(name="Value")
                    fig = px.bar(sb_df, x=comp_grp, y="Value", color=comp_sub,
                        barmode="relative", template=TPL,
                        title=f"100% Stacked: {comp_grp} by {comp_sub}",
                        color_discrete_sequence=COLORS)
                    fig.update_xaxes(tickangle=-35)
                else:
                    st.warning("Select a Sub-group for 100% Stacked Bar.")
                    fig = px.bar(base, x=comp_grp, y="Value", template=TPL)

            st.plotly_chart(std_layout(fig, h=460), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 5️⃣ TIME SERIES
        # ═══════════════════════════════════════════════════════════════
        elif "Time Series" in active_section:
            st.markdown("#### 5️⃣ Time Series")
            if not date_cols:
                st.info("No date/time columns detected. Columns with 'date','time','month','year' in name are auto-detected.")
            elif not num_cols:
                st.info("No numeric columns found.")
            else:
                t1,t2,t3 = st.columns(3)
                with t1:
                    ts_date  = st.selectbox("Date column", date_cols, key="ts_date2")
                    ts_val   = st.selectbox("Value column", num_cols, key="ts_val2")
                    ts_grp   = st.selectbox("Group by", ["None"]+cat_cols, key="ts_grp")
                with t2:
                    ts_freq  = st.selectbox("Resample", ["None","Day","Week","Month","Quarter","Year"], key="ts_freq2")
                    ts_agg   = st.selectbox("Aggregation", ["Sum","Mean","Count","Max","Min"], key="ts_agg2")
                    ts_type  = st.selectbox("Chart type", ["Line","Area","Bar","Scatter","Step"], key="ts_type")
                with t3:
                    ts_roll  = st.slider("Rolling average (0=off)", 0, 30, 0, key="ts_roll")
                    ts_annot = st.checkbox("Annotate max/min", True, key="ts_annot")

                try:
                    ts_df = df[[ts_date, ts_val] + ([ts_grp] if ts_grp!="None" else [])].copy()
                    ts_df[ts_date] = pd.to_datetime(ts_df[ts_date], errors="coerce")
                    ts_df = ts_df.dropna(subset=[ts_date])

                    freq_map2 = {"Day":"D","Week":"W","Month":"ME","Quarter":"QE","Year":"YE"}
                    agg_fn = ts_agg.lower()

                    if ts_grp != "None":
                        if ts_freq != "None":
                            ts_df = ts_df.set_index(ts_date)
                            ts_df = ts_df.groupby(ts_grp).resample(freq_map2[ts_freq])[ts_val].agg(agg_fn).reset_index()
                        plot_kw = dict(data_frame=ts_df, x=ts_date, y=ts_val, color=ts_grp,
                            template=TPL, title=f"{ts_val} over Time by {ts_grp}",
                            color_discrete_sequence=COLORS)
                    else:
                        ts_df = ts_df[[ts_date, ts_val]].sort_values(ts_date)
                        if ts_freq != "None":
                            ts_df = ts_df.set_index(ts_date).resample(freq_map2[ts_freq])[ts_val].agg(agg_fn).reset_index()
                        if ts_roll > 0:
                            ts_df[f"Rolling {ts_roll}"] = ts_df[ts_val].rolling(ts_roll).mean()
                        plot_kw = dict(data_frame=ts_df, x=ts_date, y=ts_val,
                            template=TPL, title=f"{ts_agg} of {ts_val} over Time",
                            color_discrete_sequence=["#3b82f6"])

                    if ts_type == "Line":
                        fig = px.line(**plot_kw)
                        fig.update_traces(line_width=2)
                    elif ts_type == "Area":
                        fig = px.area(**plot_kw)
                    elif ts_type == "Bar":
                        fig = px.bar(**plot_kw)
                    elif ts_type == "Scatter":
                        fig = px.scatter(**plot_kw)
                    else:  # Step
                        fig = px.line(**plot_kw, line_shape="hv")

                    # Rolling avg overlay
                    if ts_roll > 0 and ts_grp == "None" and f"Rolling {ts_roll}" in ts_df.columns:
                        fig.add_scatter(x=ts_df[ts_date], y=ts_df[f"Rolling {ts_roll}"],
                            mode="lines", name=f"{ts_roll}-period avg",
                            line=dict(color="red", dash="dash", width=2))

                    # Annotate max/min
                    if ts_annot and ts_grp == "None":
                        max_row = ts_df.loc[ts_df[ts_val].idxmax()]
                        min_row = ts_df.loc[ts_df[ts_val].idxmin()]
                        fig.add_annotation(x=max_row[ts_date], y=max_row[ts_val],
                            text=f"Max: {max_row[ts_val]:.1f}", showarrow=True, arrowhead=2,
                            bgcolor="white", bordercolor="#3b82f6")
                        fig.add_annotation(x=min_row[ts_date], y=min_row[ts_val],
                            text=f"Min: {min_row[ts_val]:.1f}", showarrow=True, arrowhead=2,
                            bgcolor="white", bordercolor="#ef4444")

                    st.plotly_chart(std_layout(fig, h=450), use_container_width=True)
                except Exception as e:
                    st.error(f"Time series error: {e}")

        # ═══════════════════════════════════════════════════════════════
        # 6️⃣ RANKING
        # ═══════════════════════════════════════════════════════════════
        elif "Ranking" in active_section:
            st.markdown("#### 6️⃣ Ranking")
            rk1,rk2 = st.columns(2)
            with rk1:
                rk_grp  = st.selectbox("Rank by category", cat_cols if cat_cols else all_cols, key="rk_grp")
                rk_val  = st.selectbox("Value", ["Row Count"]+num_cols, key="rk_val")
                rk_agg  = st.selectbox("Aggregation", ["Count","Sum","Mean","Median","Max"], key="rk_agg")
            with rk2:
                rk_n    = st.slider("Top / Bottom N", 3, 50, 15, key="rk_n")
                rk_show = st.radio("Show", ["Top N","Bottom N","Both"], key="rk_show")
                rk_pal  = st.selectbox("Palette", PALETTES, key="rk_pal")

            agg_fn3 = {"Count":"count","Sum":"sum","Mean":"mean","Median":"median","Max":"max"}
            if rk_val == "Row Count":
                full_rank = df[rk_grp].fillna("(blank)").astype(str).value_counts().reset_index()
                full_rank.columns=[rk_grp,"Value"]
            else:
                full_rank = (df.groupby(df[rk_grp].fillna("(blank)").astype(str))[rk_val]
                    .agg(agg_fn3[rk_agg]).round(2).reset_index().sort_values(rk_val, ascending=False))
                full_rank.columns=[rk_grp,"Value"]

            full_rank = full_rank.sort_values("Value", ascending=False)
            if rk_show == "Top N":
                rk_df = full_rank.head(rk_n)
                rk_df["Rank"] = range(1, len(rk_df)+1)
                title6 = f"Top {rk_n}"
            elif rk_show == "Bottom N":
                rk_df = full_rank.tail(rk_n).sort_values("Value")
                rk_df["Rank"] = range(len(full_rank)-len(rk_df)+1, len(full_rank)+1)
                title6 = f"Bottom {rk_n}"
            else:
                top = full_rank.head(rk_n).copy(); top["Group"]="Top"
                bot = full_rank.tail(rk_n).sort_values("Value").copy(); bot["Group"]="Bottom"
                rk_df = pd.concat([top, bot])
                title6 = f"Top & Bottom {rk_n}"

            fig = px.bar(rk_df, y=rk_grp, x="Value", orientation="h",
                color="Value", color_continuous_scale=rk_pal,
                title=f"{title6} — {rk_agg} of {rk_val} by {rk_grp}", template=TPL,
                text_auto=True)
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(std_layout(fig, h=max(400, rk_n*28)), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 7️⃣ CORRELATION
        # ═══════════════════════════════════════════════════════════════
        elif "Correlation" in active_section:
            st.markdown("#### 7️⃣ Correlation")
            corr_options = ["Heatmap"]
            if SCIPY_OK:
                corr_options.append("Clustermap")
            corr_options.append("Bar (top pairs)")
            corr_type = st.selectbox("Chart", corr_options, key="corr_type")

            if len(num_cols) < 2:
                st.info("Select at least 2 numeric columns to view correlation charts.")
                corr_cols = num_cols
                corr_meth = "pearson"
                corr_pal = "RdBu"
            else:
                cr1, cr2 = st.columns(2)
                with cr1:
                    corr_cols = st.multiselect("Columns to include", num_cols, default=num_cols[:min(8, len(num_cols))], key="corr_cols")
                    corr_meth = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_meth")
                with cr2:
                    corr_pal = st.selectbox("Palette", ["RdBu", "RdYlGn", "Viridis", "Plasma", "Blues"], key="corr_pal")

                if len(corr_cols) >= 2:
                    corr = df[corr_cols].corr(method=corr_meth).round(3)
                    if corr_type == "Heatmap":
                        fig = go.Figure(go.Heatmap(
                            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                            colorscale=corr_pal, zmid=0,
                            text=corr.values.round(2), texttemplate="%{text}",
                            textfont={"size":11}, hoverongaps=False,
                        ))
                        fig.update_layout(title=f"{corr_meth.title()} Correlation Heatmap", template=TPL,
                            height=max(380, len(corr_cols)*50))
                    elif corr_type == "Clustermap":
                        if SCIPY_OK:
                            dist = 1 - corr.abs()
                            dist = dist.clip(lower=0)
                            np.fill_diagonal(dist.values, 0)
                            try:
                                link = linkage(squareform(dist.values), method="ward")
                                order = leaves_list(link)
                                corr_r = corr.iloc[order, order]
                                title = "Clustered Correlation Heatmap"
                            except Exception:
                                corr_r = corr
                                title = "Clustered Correlation Heatmap"
                        else:
                            corr_r = corr
                            title = "Correlation Heatmap"
                            st.warning("Scipy is required for clustermap. Install scipy or choose another chart.")
                        fig = go.Figure(go.Heatmap(
                            z=corr_r.values, x=corr_r.columns.tolist(), y=corr_r.columns.tolist(),
                            colorscale=corr_pal, zmid=0,
                            text=corr_r.values.round(2), texttemplate="%{text}",
                        ))
                        fig.update_layout(title=title, template=TPL,
                            height=max(380, len(corr_cols)*50))
                    else:  # Bar top pairs
                        pairs = []
                        cols_list = corr_cols
                        for i in range(len(cols_list)):
                            for j in range(i+1, len(cols_list)):
                                pairs.append({"Pair": f"{cols_list[i]} × {cols_list[j]}",
                                    "Correlation": corr.loc[cols_list[i], cols_list[j]]})
                        pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
                        fig = px.bar(pairs_df, x="Pair", y="Correlation",
                            color="Correlation", color_continuous_scale="RdBu",
                            title=f"Top Correlations ({corr_meth})", template=TPL)
                        fig.update_xaxes(tickangle=-45)
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")

                    st.plotly_chart(std_layout(fig), use_container_width=True)
                else:
                    st.info("Select at least 2 columns.")

        # ═══════════════════════════════════════════════════════════════
        # 8️⃣ MULTI-GROUP COMPARISON
        # ═══════════════════════════════════════════════════════════════
        elif "Multi-Group" in active_section:
            st.markdown("#### 8️⃣ Multi-Group Comparison")
            mg1,mg2 = st.columns(2)
            with mg1:
                mg_x   = st.selectbox("X (primary group)", cat_cols if cat_cols else all_cols, key="mg_x")
                mg_col = st.selectbox("Color group", ["None"]+cat_cols, key="mg_col")
                mg_val = st.selectbox("Value", ["Row Count"]+num_cols, key="mg_val")
                mg_agg = st.selectbox("Aggregation", ["Count","Sum","Mean","Median"], key="mg_agg")
            with mg2:
                mg_type= st.selectbox("Chart type", ["Grouped Bar","Stacked Bar","100% Stacked","Heatmap Table","Radar","Parallel Categories"], key="mg_type")
                mg_topn= st.slider("Top N per group", 3, 20, 8, key="mg_topn")

            agg_fn4 = {"Count":"count","Sum":"sum","Mean":"mean","Median":"median"}
            if mg_val == "Row Count":
                if mg_col != "None" and mg_col in df.columns:
                    mg_df = df.groupby([mg_x, mg_col]).size().reset_index(name="Value")
                else:
                    mg_df = df[mg_x].value_counts().head(mg_topn).reset_index()
                    mg_df.columns = [mg_x, "Value"]
            else:
                grp_list = [mg_x] + ([mg_col] if mg_col != "None" else [])
                mg_df = df.groupby([df[c].fillna("(blank)").astype(str) for c in grp_list])[mg_val].agg(agg_fn4[mg_agg]).round(2).reset_index()
                mg_df.columns = grp_list + ["Value"]

            color_c = mg_col if mg_col != "None" else None
            label8 = f"{mg_agg} of {mg_val}" if mg_val != "Row Count" else "Count"

            if mg_type == "Grouped Bar":
                fig = px.bar(mg_df, x=mg_x, y="Value", color=color_c,
                    barmode="group", template=TPL, color_discrete_sequence=COLORS,
                    title=f"Grouped: {label8} by {mg_x}" + (f" & {mg_col}" if color_c else ""),
                    text_auto=True)
                fig.update_xaxes(tickangle=-35)
            elif mg_type == "Stacked Bar":
                fig = px.bar(mg_df, x=mg_x, y="Value", color=color_c,
                    barmode="stack", template=TPL, color_discrete_sequence=COLORS,
                    title=f"Stacked: {label8} by {mg_x}")
                fig.update_xaxes(tickangle=-35)
            elif mg_type == "100% Stacked":
                fig = px.bar(mg_df, x=mg_x, y="Value", color=color_c,
                    barmode="relative", template=TPL, color_discrete_sequence=COLORS,
                    title=f"100% Stacked: {label8}")
                fig.update_xaxes(tickangle=-35)
            elif mg_type == "Heatmap Table":
                if color_c:
                    piv = mg_df.pivot_table(index=mg_x, columns=mg_col, values="Value", aggfunc="sum").fillna(0)
                    fig = go.Figure(go.Heatmap(
                        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
                        colorscale="Blues", text=piv.values.round(1),
                        texttemplate="%{text}", textfont={"size":10},
                    ))
                    fig.update_layout(title=f"{label8}: {mg_x} × {mg_col}", template=TPL)
                else:
                    st.warning("Select a Color group for Heatmap Table.")
                    fig = go.Figure()
            elif mg_type == "Radar":
                if color_c and color_c in df.columns:
                    cats = mg_df[mg_x].unique().tolist()[:12]
                    groups = mg_df[mg_col].unique().tolist()[:6]
                    fig = go.Figure()
                    for grp in groups:
                        sub = mg_df[mg_df[mg_col]==grp]
                        vals = [sub.loc[sub[mg_x]==c,"Value"].sum() for c in cats]
                        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                            fill="toself", name=str(grp)))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                        title=f"Radar: {label8}", template=TPL)
                else:
                    st.warning("Select a Color group for Radar chart.")
                    fig = go.Figure()
            else:  # Parallel Categories
                pc_cols = [mg_x] + ([mg_col] if color_c else [])
                fig = px.parallel_categories(df[pc_cols+([mg_val] if mg_val!="Row Count" else [])].dropna().head(2000),
                    dimensions=pc_cols, template=TPL,
                    title=f"Parallel Categories: {' → '.join(pc_cols)}")

            st.plotly_chart(std_layout(fig, h=480), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 9️⃣ CUSTOM CHART BUILDER
        # ═══════════════════════════════════════════════════════════════
        elif "Custom" in active_section:
            st.markdown("#### 9️⃣ Custom Chart Builder")
            st.caption("Full control — pick any columns and any chart type.")
            cu1,cu2,cu3 = st.columns(3)
            with cu1:
                cu_x     = st.selectbox("X axis", ["None"]+all_cols, key="cu_x")
                cu_y     = st.selectbox("Y axis", ["None"]+all_cols, key="cu_y")
                cu_color = st.selectbox("Color", ["None"]+all_cols, key="cu_color")
                cu_size  = st.selectbox("Size", ["None"]+num_cols, key="cu_size")
            with cu2:
                cu_facet = st.selectbox("Facet column", ["None"]+cat_cols, key="cu_facet")
                cu_facr  = st.selectbox("Facet row", ["None"]+cat_cols, key="cu_facr")
                cu_type  = st.selectbox("Chart type", [
                    "Bar","Horizontal Bar","Line","Area","Scatter","Bubble",
                    "Box","Violin","Strip","Histogram","Pie","Donut",
                    "Treemap","Sunburst","Funnel","ECDF","Density Heatmap",
                    "Parallel Coordinates","Parallel Categories","Scatter Matrix"
                ], key="cu_type")
            with cu3:
                cu_agg   = st.selectbox("Aggregate Y by X?", ["None","Sum","Mean","Count","Median","Max","Min"], key="cu_agg")
                cu_pal   = st.selectbox("Color palette", PALETTES+["Plotly","D3","G10","T10"], key="cu_pal")
                cu_n     = st.slider("Sample rows", 100, len(df), min(2000, len(df)), 100, key="cu_n")
                cu_h     = st.slider("Chart height", 300, 900, 450, 50, key="cu_h")
                cu_tl    = st.selectbox("Trendline", ["None","OLS","Lowess"], key="cu_tl")

            plot_df2 = df.sample(cu_n)
            x_c = cu_x if cu_x != "None" else None
            y_c = cu_y if cu_y != "None" else None
            color_c2 = cu_color if cu_color != "None" else None
            size_c   = cu_size if cu_size != "None" else None
            facet_c  = cu_facet if cu_facet != "None" else None
            facr_c   = cu_facr if cu_facr != "None" else None

            # Pre-aggregate if requested
            if cu_agg != "None" and x_c and y_c:
                agg_fn5 = {"Sum":"sum","Mean":"mean","Count":"count","Median":"median","Max":"max","Min":"min"}
                plot_df2 = (df.groupby(df[x_c].fillna("(blank)").astype(str))[y_c]
                    .agg(agg_fn5[cu_agg]).round(2).reset_index())
                plot_df2.columns = [x_c, y_c]

            pal_map = {p: getattr(px.colors.sequential, p, COLORS) for p in PALETTES}
            pal_map.update({"Plotly":px.colors.qualitative.Plotly,"D3":px.colors.qualitative.D3,
                            "G10":px.colors.qualitative.G10,"T10":px.colors.qualitative.T10})
            disc_pal = pal_map.get(cu_pal, COLORS)
            tl_arg = {"OLS":"ols","Lowess":"lowess","None":None}[cu_tl]

            try:
                kw2 = dict(data_frame=plot_df2, template=TPL,
                    title=f"{cu_type}: {y_c or ''} by {x_c or ''}")
                if x_c: kw2["x"] = x_c
                if y_c: kw2["y"] = y_c
                if color_c2: kw2["color"] = color_c2
                if facet_c: kw2["facet_col"] = facet_c; kw2["facet_col_wrap"] = 3
                if facr_c: kw2["facet_row"] = facr_c

                if cu_type == "Bar":
                    fig = px.bar(**kw2, color_discrete_sequence=disc_pal, text_auto=True)
                    if x_c: fig.update_xaxes(tickangle=-35)
                elif cu_type == "Horizontal Bar":
                    if x_c and y_c:
                        kw2["x"], kw2["y"] = y_c, x_c
                    fig = px.bar(**kw2, orientation="h", color_discrete_sequence=disc_pal)
                elif cu_type == "Line":
                    fig = px.line(**kw2, color_discrete_sequence=disc_pal)
                elif cu_type == "Area":
                    fig = px.area(**kw2, color_discrete_sequence=disc_pal)
                elif cu_type in ("Scatter","Bubble"):
                    if tl_arg: kw2["trendline"] = tl_arg
                    if size_c and cu_type=="Bubble": kw2["size"]=size_c; kw2["size_max"]=25
                    fig = px.scatter(**kw2, color_discrete_sequence=disc_pal, opacity=0.7)
                elif cu_type == "Box":
                    fig = px.box(**kw2, color_discrete_sequence=disc_pal, points="outliers")
                elif cu_type == "Violin":
                    fig = px.violin(**kw2, color_discrete_sequence=disc_pal, box=True)
                elif cu_type == "Strip":
                    fig = px.strip(**kw2, color_discrete_sequence=disc_pal)
                elif cu_type == "Histogram":
                    fig = px.histogram(**kw2, color_discrete_sequence=disc_pal, nbins=40, marginal="rug")
                elif cu_type in ("Pie","Donut"):
                    fig = px.pie(plot_df2, names=x_c, values=y_c, hole=0.4 if cu_type=="Donut" else 0,
                        color_discrete_sequence=disc_pal, title=kw2["title"])
                elif cu_type == "Treemap":
                    path = [p for p in [x_c, color_c2] if p]
                    fig = px.treemap(plot_df2, path=path, values=y_c,
                        color_continuous_scale=cu_pal, title=kw2["title"])
                elif cu_type == "Sunburst":
                    path = [p for p in [x_c, color_c2] if p]
                    fig = px.sunburst(plot_df2, path=path, values=y_c,
                        color_discrete_sequence=disc_pal, title=kw2["title"])
                elif cu_type == "Funnel":
                    fig = px.funnel(**kw2, color_discrete_sequence=disc_pal)
                elif cu_type == "ECDF":
                    fig = px.ecdf(**kw2, color_discrete_sequence=disc_pal)
                elif cu_type == "Density Heatmap":
                    fig = px.density_heatmap(plot_df2, x=x_c, y=y_c,
                        nbinsx=30, nbinsy=30, color_continuous_scale=cu_pal, title=kw2["title"])
                elif cu_type == "Parallel Coordinates":
                    pc_num = [c for c in [x_c,y_c,size_c] if c and pd.api.types.is_numeric_dtype(df[c])]
                    if len(pc_num) >= 2:
                        fig = px.parallel_coordinates(plot_df2, dimensions=pc_num,
                            color=pc_num[0], color_continuous_scale=cu_pal, title=kw2["title"])
                    else:
                        st.warning("Parallel Coordinates needs numeric columns for X, Y.")
                        fig = go.Figure()
                elif cu_type == "Parallel Categories":
                    pc_c = [c for c in [x_c, color_c2] if c]
                    fig = px.parallel_categories(plot_df2, dimensions=pc_c, title=kw2["title"])
                elif cu_type == "Scatter Matrix":
                    sm_cols = [c for c in [x_c,y_c,size_c] if c]
                    if len(sm_cols) < 2: sm_cols = num_cols[:4]
                    fig = px.scatter_matrix(plot_df2, dimensions=sm_cols, color=color_c2,
                        color_discrete_sequence=disc_pal, title=kw2["title"], opacity=0.6)
                    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
                else:
                    fig = go.Figure()

                st.plotly_chart(std_layout(fig, h=cu_h), use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        # ═══════════════════════════════════════════════════════════════
        # 🔟 MISSING DATA MAP
        # ═══════════════════════════════════════════════════════════════
        elif "Missing" in active_section:
            st.markdown("#### 🔟 Missing Data Map")
            miss_cols = st.multiselect("Columns to inspect", all_cols, default=all_cols, key="miss_cols")
            miss_n    = st.slider("Sample rows", 50, 500, 100, key="miss_n")

            if miss_cols:
                sub = df[miss_cols].isnull().astype(int).sample(min(miss_n, len(df))).reset_index(drop=True)
                fig = go.Figure(go.Heatmap(
                    z=sub.T.values,
                    x=[str(i) for i in sub.index],
                    y=miss_cols,
                    colorscale=[[0,"#d1fae5"],[1,"#ef4444"]],
                    showscale=True,
                    colorbar=dict(title="", tickvals=[0,1], ticktext=["Present","Missing"]),
                ))
                fig.update_layout(title=f"Missing Data (sample {len(sub)} rows)", template=TPL,
                    height=max(300, len(miss_cols)*24),
                    margin=dict(t=45,b=20,l=130,r=20),
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(std_layout(fig), use_container_width=True)

                st.markdown("**Missing count per column:**")
                miss_summary = df[miss_cols].isnull().sum().reset_index()
                miss_summary.columns=["Column","Missing"]
                miss_summary["%"]=( miss_summary["Missing"]/len(df)*100).round(1)
                miss_summary = miss_summary.sort_values("Missing",ascending=False)
                fig2 = px.bar(miss_summary, x="Column", y="Missing", color="%",
                    color_continuous_scale="Reds", template=TPL,
                    title="Missing Values per Column", text_auto=True)
                fig2.update_xaxes(tickangle=-35)
                st.plotly_chart(std_layout(fig2), use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # 1️⃣1️⃣ PAIR PLOT
        # ═══════════════════════════════════════════════════════════════
        elif "Pair" in active_section:
            st.markdown("#### 1️⃣1️⃣ Pair Plot")
            pp1,pp2 = st.columns(2)
            with pp1:
                pair_cols = st.multiselect("Columns (2–8)", num_cols,
                    default=num_cols[:min(4,len(num_cols))], key="pair_cols2")
                pair_color = st.selectbox("Color by", ["None"]+cat_cols, key="pair_color2")
            with pp2:
                pair_n    = st.slider("Sample rows", 100, 2000, 500, key="pair_n")
                pair_diag = st.selectbox("Diagonal", ["histogram","box","violin"], key="pair_diag")

            if len(pair_cols) >= 2:
                pair_df = df[pair_cols+([pair_color] if pair_color!="None" else [])].dropna().sample(min(pair_n,len(df)))
                kw_p = dict(data_frame=pair_df, dimensions=pair_cols, template=TPL,
                    title="Pair Plot", opacity=0.55)
                if pair_color != "None":
                    kw_p["color"] = pair_color
                    kw_p["color_discrete_sequence"] = COLORS
                fig = px.scatter_matrix(**kw_p)
                fig.update_traces(diagonal_visible=(pair_diag=="histogram"),
                    marker=dict(size=3))
                fig.update_layout(height=650, margin=dict(t=45,b=20,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least 2 numeric columns.")
