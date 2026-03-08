"""
Fraud Detection Command Center — Streamlit Dashboard
=====================================================
Phase 10: THE SENTINEL (Human-in-the-Loop)

Usage:
    # Terminal 1: python -m src.api.inference_api
    # Terminal 2: streamlit run src/dashboard/app.py --server.port 8520
"""

import os, sys, json, math, time, requests
import streamlit as st
import streamlit.components.v1 as components

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.logger import load_config, get_logger, resolve_path

config = load_config()
logger = get_logger("phase10.dashboard")
API_URL = config["dashboard"]["api_url"]
DASHBOARD_TITLE = config["dashboard"]["title"]

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title=DASHBOARD_TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root { --primary-navy:#0a1628; --secondary-navy:#1a2942; --accent-blue:#2563eb; --light-blue:#3b82f6; --ice-blue:#dbeafe; --pale-blue:#eff6ff; --steel-gray:#64748b; --light-gray:#f1f5f9; --border-gray:#e2e8f0; --white:#ffffff; --success-green:#059669; --warning-amber:#d97706; --danger-red:#dc2626; --text-primary:#0f172a; --text-secondary:#475569; --text-muted:#94a3b8; }
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header[data-testid="stHeader"] {background:transparent;}

    .top-navbar { background:linear-gradient(135deg,var(--primary-navy) 0%,var(--secondary-navy) 100%); padding:0.8rem 2rem; margin:-1rem -1rem 1.5rem -1rem; display:flex; align-items:center; justify-content:space-between; border-bottom:3px solid var(--accent-blue); box-shadow:0 4px 20px rgba(10,22,40,0.3); }
    .navbar-brand { display:flex; align-items:center; gap:0.75rem; }
    .navbar-logo { font-size:1.6rem; }
    .navbar-title { font-size:1.35rem; font-weight:700; color:#fff; letter-spacing:-0.02em; }
    .navbar-subtitle { font-size:0.72rem; color:var(--text-muted); letter-spacing:0.08em; text-transform:uppercase; margin-top:0.1rem; }
    .navbar-status { display:flex; align-items:center; gap:1.5rem; }
    .navbar-status-item { display:flex; align-items:center; gap:0.4rem; color:var(--text-muted); font-size:0.78rem; font-weight:500; }
    .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
    .status-dot.online { background-color:var(--success-green); box-shadow:0 0 8px rgba(5,150,105,0.6); }
    .status-dot.offline { background-color:var(--danger-red); box-shadow:0 0 8px rgba(220,38,38,0.6); }

    [data-testid="stSidebar"] { background:linear-gradient(180deg,var(--primary-navy) 0%,#0d1f3c 100%); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color:#c8d6e5 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 { color:#fff !important; }
    [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.1); }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(135deg,#1e40af 0%,#2563eb 100%) !important; color:#fff !important; border:none !important; font-weight:600 !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 100%) !important; }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) { background:rgba(255,255,255,0.08) !important; color:#c8d6e5 !important; border:1px solid rgba(255,255,255,0.15) !important; font-weight:500 !important; }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover { background:rgba(255,255,255,0.14) !important; }

    .kpi-card { background:var(--white); border:1px solid var(--border-gray); border-radius:12px; padding:1.25rem 1.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.04); transition:box-shadow 0.2s ease,transform 0.2s ease; position:relative; overflow:hidden; height:100%; }
    .kpi-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08); transform:translateY(-1px); }
    .kpi-card::before { content:''; position:absolute; top:0; left:0; width:4px; height:100%; }
    .kpi-card.blue::before { background:var(--accent-blue); } .kpi-card.green::before { background:var(--success-green); }
    .kpi-card.amber::before { background:var(--warning-amber); } .kpi-card.navy::before { background:var(--primary-navy); }
    .kpi-label { font-size:0.73rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; }
    .kpi-value { font-size:1.7rem; font-weight:800; color:var(--text-primary); line-height:1.2; }
    .kpi-sub { font-size:0.72rem; color:var(--text-secondary); margin-top:0.35rem; }

    .risk-badge { display:inline-block; padding:0.2rem 0.65rem; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:0.04em; }
    .risk-badge.low { background:#d1fae5; color:#065f46; } .risk-badge.medium { background:#fef3c7; color:#92400e; }
    .risk-badge.high { background:#fed7aa; color:#9a3412; } .risk-badge.critical { background:#fecaca; color:#991b1b; }

    .section-header { font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:1.5rem 0 0.75rem 0; padding-bottom:0.5rem; border-bottom:2px solid var(--accent-blue); display:flex; align-items:center; gap:0.5rem; }
    .info-panel { background:var(--pale-blue); border:1px solid var(--ice-blue); border-left:4px solid var(--accent-blue); border-radius:8px; padding:1rem 1.25rem; margin:1rem 0; color:var(--text-primary); font-size:0.88rem; line-height:1.6; }
    .component-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin:0.5rem 0; }
    .component-chip { display:flex; align-items:center; gap:0.4rem; background:rgba(255,255,255,0.06); padding:0.4rem 0.6rem; border-radius:6px; font-size:0.76rem; color:#c8d6e5; }

    .preview-card { background:var(--white); border:1px solid var(--border-gray); border-radius:12px; padding:1.25rem; box-shadow:0 1px 4px rgba(0,0,0,0.04); max-height:520px; overflow-y:auto; }
    .preview-card h4 { margin:0 0 0.75rem 0; font-size:0.95rem; font-weight:700; color:var(--text-primary); }
    .preview-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
    .preview-table th { text-align:left; padding:0.4rem 0.6rem; background:var(--light-gray); color:var(--text-secondary); font-weight:600; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.04em; border-bottom:2px solid var(--border-gray); position:sticky; top:0; }
    .preview-table td { padding:0.35rem 0.6rem; border-bottom:1px solid var(--border-gray); color:var(--text-primary); }
    .preview-table tr:hover td { background:var(--pale-blue); }
    .preview-table .feat-name { font-weight:600; color:var(--accent-blue); font-size:0.76rem; }
    .preview-table .feat-val { font-family:'SF Mono','Fira Code',monospace; font-size:0.76rem; }
    .preview-table .feat-val.nonzero { color:var(--accent-blue); font-weight:600; }
    .preview-table .feat-group { background:var(--primary-navy); color:#fff; font-weight:600; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; padding:0.45rem 0.6rem; }

    .input-card-header { font-size:0.95rem; font-weight:700; color:var(--text-primary); margin-bottom:0.75rem; display:flex; align-items:center; gap:0.4rem; padding-bottom:0.5rem; border-bottom:2px solid var(--accent-blue); }

    .gauge-wrapper { background:var(--white); border:1px solid var(--border-gray); border-radius:16px; padding:1.5rem 1rem 1rem 1rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); text-align:center; }
    .gauge-title { font-size:0.78rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem; }
    .gauge-svg-container { display:flex; justify-content:center; }
    .gauge-value-large { font-size:2.8rem; font-weight:800; line-height:1; margin-top:-0.3rem; }
    .gauge-risk-label { display:inline-block; padding:0.3rem 1.1rem; border-radius:25px; font-size:0.85rem; font-weight:700; letter-spacing:0.05em; margin-top:0.6rem; }
    .gauge-decision-row { display:flex; justify-content:center; gap:1.5rem; margin-top:0.8rem; font-size:0.78rem; color:var(--text-secondary); }
    .gauge-decision-item { display:flex; align-items:center; gap:0.3rem; }

    .app-footer { text-align:center; padding:1.5rem 0 0.5rem 0; color:var(--text-muted); font-size:0.75rem; border-top:1px solid var(--border-gray); margin-top:2rem; }

    .stTabs [data-baseweb="tab-list"] { gap:0; background:var(--white); border:1px solid var(--border-gray); border-radius:10px; padding:0.25rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
    .stTabs [data-baseweb="tab"] { border-radius:8px; font-weight:600; font-size:0.85rem; padding:0.6rem 1.2rem; color:var(--text-secondary); }
    .stTabs [aria-selected="true"] { background:var(--accent-blue) !important; color:#fff !important; box-shadow:0 2px 8px rgba(37,99,235,0.3); }
    .stTabs [data-baseweb="tab-highlight"] { display:none; }
    .stTabs [data-baseweb="tab-border"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception: return None

def get_sample_transaction():
    try:
        r = requests.get(f"{API_URL}/sample", timeout=5)
        return r.json().get("features", {}) if r.status_code == 200 else None
    except Exception: return None

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception: return None

def load_metrics():
    """Load models/metrics.json for detailed model evaluation data."""
    try:
        path = resolve_path(config["model"]["metrics_path"])
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")
    return None

def load_comparison():
    """Load models/model_comparison.json for head-to-head results."""
    try:
        path = resolve_path(config["model"]["comparison_path"])
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load comparison: {e}")
    return None

def render_comparison_table(comparison):
    """Render XGBoost vs LightGBM comparison table via components.html."""
    if not comparison:
        return

    xgb = comparison.get("xgboost", {})
    lgb = comparison.get("lightgbm", {})
    winner = comparison.get("winner", "?")

    metrics_rows = [
        ("AUPRC (PRIMARY)", f'{xgb.get("auprc",0):.6f}', f'{lgb.get("auprc",0):.6f}', "auprc"),
        ("ROC-AUC", f'{xgb.get("roc_auc",0):.6f}', f'{lgb.get("roc_auc",0):.6f}', "roc_auc"),
        ("Precision (default)", f'{xgb.get("precision_default",0):.6f}', f'{lgb.get("precision_default",0):.6f}', "precision"),
        ("Recall (default)", f'{xgb.get("recall_default",0):.6f}', f'{lgb.get("recall_default",0):.6f}', "recall"),
        ("F1 Score (default)", f'{xgb.get("f1_default",0):.6f}', f'{lgb.get("f1_default",0):.6f}', "f1"),
        ("Optimal Threshold", f'{xgb.get("optimal_threshold",0):.6f}', f'{lgb.get("optimal_threshold",0):.6f}', "threshold"),
        ("F1 (Optimal)", f'{xgb.get("f1_optimal",0):.6f}', f'{lgb.get("f1_optimal",0):.6f}', "f1opt"),
        ("Training Time", f'{xgb.get("training_time_seconds",0):.2f}s', f'{lgb.get("training_time_seconds",0):.2f}s', "time"),
        ("Inference (ms/100)", f'{xgb.get("inference_time_ms",0):.2f}', f'{lgb.get("inference_time_ms",0):.2f}', "inference"),
    ]

    rows_html = ""
    for label, xv, lv, key in metrics_rows:
        xgb_style = ""
        lgb_style = ""
        if key == "auprc":
            if winner == "XGBoost":
                xgb_style = 'style="color:#059669;font-weight:800;"'
            else:
                lgb_style = 'style="color:#059669;font-weight:800;"'
            label = f'<strong>{label}</strong>'

        row_bg = "#f8fafc" if metrics_rows.index((label.replace("<strong>","").replace("</strong>",""), xv, lv, key)) % 2 == 0 else "#fff"
        rows_html += f"""
        <tr style="background:{row_bg};">
            <td style="padding:10px 14px;font-weight:600;color:#0f172a;font-size:0.82rem;border-bottom:1px solid #e2e8f0;">{label}</td>
            <td style="padding:10px 14px;text-align:center;font-family:monospace;font-size:0.82rem;border-bottom:1px solid #e2e8f0;" {xgb_style}>{xv}</td>
            <td style="padding:10px 14px;text-align:center;font-family:monospace;font-size:0.82rem;border-bottom:1px solid #e2e8f0;" {lgb_style}>{lv}</td>
        </tr>"""

    xgb_header_style = 'background:#059669;color:#fff;' if winner == "XGBoost" else 'background:#1a2942;color:#fff;'
    lgb_header_style = 'background:#059669;color:#fff;' if winner == "LightGBM" else 'background:#1a2942;color:#fff;'

    full_html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:'Inter',sans-serif;background:#fff;padding:16px;}}</style>
    </head><body>
    <table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">
        <thead>
            <tr>
                <th style="padding:12px 14px;text-align:left;background:#f1f5f9;color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;border-bottom:2px solid #e2e8f0;">Metric</th>
                <th style="padding:12px 14px;text-align:center;{xgb_header_style}font-size:0.82rem;font-weight:700;border-bottom:2px solid #e2e8f0;">
                    XGBoost {"🏆" if winner=="XGBoost" else ""}
                </th>
                <th style="padding:12px 14px;text-align:center;{lgb_header_style}font-size:0.82rem;font-weight:700;border-bottom:2px solid #e2e8f0;">
                    LightGBM {"🏆" if winner=="LightGBM" else ""}
                </th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <div style="margin-top:12px;padding:10px 14px;background:#d1fae5;border-radius:8px;border:1px solid #a7f3d0;font-size:0.82rem;color:#065f46;">
        🏆 <strong>Winner: {winner}</strong> — Selected by AUPRC (the correct metric for imbalanced financial data). Standard accuracy is forbidden.
    </div>
    </body></html>"""

    components.html(full_html, height=len(metrics_rows)*44 + 130, scrolling=False)

def predict_transaction(features_dict):
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": features_dict}, timeout=15)
        if r.status_code == 200: return r.json()
        return None
    except Exception as e:
        logger.error(f"Predict failed: {e}"); return None

def predict_lime(features_dict):
    """Call POST /predict/lime for LIME explanation."""
    try:
        r = requests.post(f"{API_URL}/predict/lime", json={"features": features_dict}, timeout=30)
        if r.status_code == 200: return r.json()
        logger.warning(f"LIME predict returned {r.status_code}")
        return None
    except Exception as e:
        logger.error(f"LIME predict failed: {e}"); return None

# ---------------------------------------------------------------------------
# PHASE 10: FEEDBACK HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def submit_feedback(features_dict, prediction_result, correction_type, notes=""):
    """Submit investigator feedback via POST /feedback."""
    try:
        payload = {
            "features": features_dict,
            "original_probability": prediction_result.get("fraud_probability", 0),
            "original_risk_level": prediction_result.get("risk_level", "UNKNOWN"),
            "original_is_flagged": prediction_result.get("is_flagged", False),
            "correction_type": correction_type,
            "investigator_notes": notes,
        }
        r = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.warning(f"Feedback submission returned {r.status_code}")
        return None
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return None


def get_feedback_history_from_api(limit=20):
    """Fetch feedback history from GET /feedback/history."""
    try:
        r = requests.get(f"{API_URL}/feedback/history", params={"limit": limit}, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def get_feedback_stats_from_api():
    """Fetch feedback stats from GET /feedback/stats."""
    try:
        r = requests.get(f"{API_URL}/feedback/stats", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def load_graph_summary():
    """Load graphs/graph_summary.json for network stats."""
    try:
        path = resolve_path(config["graph"]["summary_path"])
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load graph summary: {e}")
    return None

def load_fraud_rings():
    """Load graphs/fraud_rings.json for ring details."""
    try:
        path = resolve_path(config["graph"]["rings_path"])
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load fraud rings: {e}")
    return None

def load_edge_list():
    """Load graphs/fraud_edges.csv for building the network visualization."""
    try:
        path = resolve_path(config["graph"]["edge_list_path"])
        if path.exists():
            import csv
            edges = []
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edges.append(row)
            return edges
    except Exception as e:
        logger.warning(f"Failed to load edge list: {e}")
    return None

def build_network_viz_html(edges, rings_data):
    """Build an interactive network graph using pure inline JS + Canvas (no CDN needed)."""
    if not edges:
        return None

    ring_membership = {}
    if rings_data:
        for ring in rings_data.get("rings", []):
            rid = ring.get("ring_id", "")
            for node in ring.get("nodes", []):
                ring_membership[node] = rid

    ring_colors = {"RING_001": "#dc2626", "RING_002": "#2563eb", "RING_003": "#d97706"}
    default_color = "#94a3b8"

    all_nodes = set()
    for e in edges:
        all_nodes.add(e.get("source", ""))
        all_nodes.add(e.get("target", ""))

    node_list = sorted(all_nodes)
    node_index = {n: i for i, n in enumerate(node_list)}

    nodes_js_items = []
    for n in node_list:
        rid = ring_membership.get(n, "")
        color = ring_colors.get(rid, default_color)
        label = n.replace("TX_", "")
        ring_label = rid if rid else "Isolated"
        nodes_js_items.append(f'{{id:{node_index[n]},label:"{label}",ring:"{ring_label}",color:"{color}",x:Math.random()*800,y:Math.random()*500,vx:0,vy:0}}')

    edge_limit = min(len(edges), 300)
    edges_js_items = []
    for e in edges[:edge_limit]:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if src in node_index and tgt in node_index:
            sim = float(e.get("similarity", 0))
            edges_js_items.append(f'{{source:{node_index[src]},target:{node_index[tgt]},sim:{sim:.4f}}}')

    nodes_js = ",".join(nodes_js_items)
    edges_js = ",".join(edges_js_items)

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',Arial,sans-serif;background:#fff;overflow:hidden;}}
canvas{{border:1px solid #e2e8f0;border-radius:12px;cursor:grab;}}
canvas:active{{cursor:grabbing;}}
.legend{{display:flex;gap:16px;padding:8px 0;font-size:12px;color:#475569;}}
.legend-item{{display:flex;align-items:center;gap:5px;}}
.legend-dot{{width:12px;height:12px;border-radius:50%;display:inline-block;}}
.tooltip{{position:absolute;background:#0f172a;color:#fff;padding:6px 10px;border-radius:6px;font-size:11px;pointer-events:none;display:none;z-index:10;}}
.note{{font-size:11px;color:#94a3b8;margin-top:4px;}}
</style>
</head>
<body>
<div class="legend">
  <div class="legend-item"><span class="legend-dot" style="background:#dc2626;"></span>RING_001 (68 nodes)</div>
  <div class="legend-item"><span class="legend-dot" style="background:#2563eb;"></span>RING_002 (5 nodes)</div>
  <div class="legend-item"><span class="legend-dot" style="background:#94a3b8;"></span>Isolated</div>
</div>
<canvas id="c" width="920" height="480"></canvas>
<div class="tooltip" id="tt"></div>
<div class="note">{edge_limit} of {len(edges)} edges shown. Drag nodes to explore. Hover for details.</div>
<script>
var nodes=[{nodes_js}];
var edges=[{edges_js}];
var canvas=document.getElementById('c');
var ctx=canvas.getContext('2d');
var tt=document.getElementById('tt');
var drag=null,hoverNode=null,offsetX=0,offsetY=0;

function simulate(){{
  for(var i=0;i<nodes.length;i++){{
    for(var j=i+1;j<nodes.length;j++){{
      var dx=nodes[j].x-nodes[i].x;
      var dy=nodes[j].y-nodes[i].y;
      var d=Math.sqrt(dx*dx+dy*dy)+0.1;
      var f=200/(d*d);
      nodes[i].vx-=dx/d*f;nodes[i].vy-=dy/d*f;
      nodes[j].vx+=dx/d*f;nodes[j].vy+=dy/d*f;
    }}
  }}
  for(var i=0;i<edges.length;i++){{
    var s=nodes[edges[i].source],t=nodes[edges[i].target];
    var dx=t.x-s.x,dy=t.y-s.y;
    var d=Math.sqrt(dx*dx+dy*dy)+0.1;
    var f=(d-60)*0.003;
    s.vx+=dx/d*f;s.vy+=dy/d*f;
    t.vx-=dx/d*f;t.vy-=dy/d*f;
  }}
  for(var i=0;i<nodes.length;i++){{
    nodes[i].vx+=(460-nodes[i].x)*0.001;
    nodes[i].vy+=(240-nodes[i].y)*0.001;
    nodes[i].vx*=0.85;nodes[i].vy*=0.85;
    if(drag!==i){{nodes[i].x+=nodes[i].vx;nodes[i].y+=nodes[i].vy;}}
    nodes[i].x=Math.max(10,Math.min(910,nodes[i].x));
    nodes[i].y=Math.max(10,Math.min(470,nodes[i].y));
  }}
}}

function draw(){{
  ctx.clearRect(0,0,920,480);
  ctx.strokeStyle='rgba(200,210,220,0.3)';ctx.lineWidth=0.5;
  for(var i=0;i<edges.length;i++){{
    var s=nodes[edges[i].source],t=nodes[edges[i].target];
    ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(t.x,t.y);ctx.stroke();
  }}
  for(var i=0;i<nodes.length;i++){{
    var n=nodes[i];
    var r=n.ring!=='Isolated'?6:4;
    ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=n.color;ctx.fill();
    ctx.strokeStyle='rgba(0,0,0,0.2)';ctx.lineWidth=1;ctx.stroke();
    if(hoverNode===i){{
      ctx.beginPath();ctx.arc(n.x,n.y,r+3,0,Math.PI*2);
      ctx.strokeStyle=n.color;ctx.lineWidth=2;ctx.stroke();
    }}
  }}
}}

function tick(){{simulate();draw();requestAnimationFrame(tick);}}

for(var s=0;s<80;s++)simulate();
tick();

canvas.addEventListener('mousemove',function(e){{
  var rect=canvas.getBoundingClientRect();
  var mx=e.clientX-rect.left,my=e.clientY-rect.top;
  hoverNode=null;
  for(var i=0;i<nodes.length;i++){{
    var dx=nodes[i].x-mx,dy=nodes[i].y-my;
    if(dx*dx+dy*dy<64){{hoverNode=i;break;}}
  }}
  if(drag!==null){{nodes[drag].x=mx;nodes[drag].y=my;}}
  if(hoverNode!==null){{
    var n=nodes[hoverNode];
    tt.style.display='block';tt.style.left=(e.clientX+10)+'px';tt.style.top=(e.clientY-30)+'px';
    tt.textContent=n.label+' | Ring: '+n.ring;
  }}else{{tt.style.display='none';}}
}});
canvas.addEventListener('mousedown',function(e){{
  var rect=canvas.getBoundingClientRect();
  var mx=e.clientX-rect.left,my=e.clientY-rect.top;
  for(var i=0;i<nodes.length;i++){{
    var dx=nodes[i].x-mx,dy=nodes[i].y-my;
    if(dx*dx+dy*dy<64){{drag=i;break;}}
  }}
}});
canvas.addEventListener('mouseup',function(){{drag=null;}});
canvas.addEventListener('mouseleave',function(){{drag=null;tt.style.display='none';}});
</script>
</body>
</html>"""
    return html

def render_ring_details(rings_data):
    """Render fraud ring detail cards via components.html."""
    if not rings_data: return
    rings = rings_data.get("rings", [])
    if not rings: return

    cards = ""
    for ring in rings:
        rid = ring.get("ring_id", "?")
        size = ring.get("size", 0)
        edges = ring.get("num_edges", 0)
        density = ring.get("density", 0)
        total_amt = ring.get("total_amount", 0)
        avg_amt = ring.get("avg_amount", 0)
        time_span = ring.get("time_span_seconds", 0)
        avg_sim = ring.get("avg_similarity", 0)

        if size >= 50: border_color = "#dc2626"; severity = "HIGH RISK"
        elif size >= 10: border_color = "#d97706"; severity = "MEDIUM RISK"
        else: border_color = "#2563eb"; severity = "MONITORED"

        cards += f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-left:5px solid {border_color};border-radius:12px;padding:20px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <div>
                    <span style="font-size:1.1rem;font-weight:800;color:#0f172a;">{rid}</span>
                    <span style="margin-left:10px;padding:3px 10px;border-radius:15px;font-size:0.7rem;font-weight:700;background:{border_color}15;color:{border_color};">{severity}</span>
                </div>
                <span style="font-size:0.75rem;color:#94a3b8;">Density: {density:.2%}</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600;letter-spacing:0.04em;">Nodes</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{size}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600;letter-spacing:0.04em;">Edges</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{edges}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600;letter-spacing:0.04em;">Avg Similarity</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{avg_sim:.2%}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;font-weight:600;letter-spacing:0.04em;">Time Span</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#0f172a;">{time_span:.1f}s</div>
                </div>
            </div>
        </div>
        """

    full_html = f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"><style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:'Inter',sans-serif;background:transparent;padding:8px 0;}}</style></head><body>{cards}</body></html>"""
    components.html(full_html, height=len(rings)*160 + 20, scrolling=False)

def build_preview_html(features):
    if not features: return ""
    groups = {"Transaction Details":["Amount","Time"], "Engineered Features":["amount_log","hour_sin","hour_cos","is_night","amount_zscore","freq_1h","freq_24h","time_since_last"], "PCA Components":[f"V{i}" for i in range(1,29)]}
    html = '<div class="preview-card"><h4>📋 Transaction Preview</h4><table class="preview-table"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>'
    for gn, fl in groups.items():
        html += f'<tr><td colspan="2" class="feat-group">{gn}</td></tr>'
        for f in fl:
            v = features.get(f, 0.0); vc = "feat-val nonzero" if v != 0.0 else "feat-val"
            html += f'<tr><td class="feat-name">{f}</td><td class="{vc}">{v:.4f}</td></tr>'
    html += "</tbody></table></div>"
    return html

def build_gauge_svg(probability, risk_level):
    cx,cy,r = 120,110,90; sw=16; al=math.pi*r; fl=al*probability; el=al-fl
    cm = {"LOW":"#059669","MEDIUM":"#d97706","HIGH":"#ea580c","CRITICAL":"#dc2626"}
    gc = cm.get(risk_level,"#059669")
    return f'<svg width="240" height="140" viewBox="0 0 240 140"><path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="#e2e8f0" stroke-width="{sw}" stroke-linecap="round"/><path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="{gc}" stroke-width="{sw}" stroke-linecap="round" stroke-dasharray="{fl} {el}" style="filter:drop-shadow(0 0 6px {gc}40);"/><text x="{cx-r-5}" y="{cy+20}" font-size="11" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">0%</text><text x="{cx}" y="18" font-size="11" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">50%</text><text x="{cx+r+5}" y="{cy+20}" font-size="11" fill="#94a3b8" text-anchor="middle" font-family="Inter,sans-serif">100%</text></svg>'

def render_shap_chart(shap_explanations):
    if not shap_explanations:
        st.markdown('<div class="info-panel">No SHAP explanations available.</div>', unsafe_allow_html=True)
        return
    max_abs = max(abs(s.get("shap_value",0)) for s in shap_explanations)
    if max_abs == 0: max_abs = 1
    n_inc = sum(1 for s in shap_explanations if s.get("shap_value",0)>0)
    n_dec = sum(1 for s in shap_explanations if s.get("shap_value",0)<=0)
    rows = ""
    for item in shap_explanations:
        fn=item.get("feature_name","?"); sv=item.get("shap_value",0); fv=item.get("feature_value",0)
        bp = (abs(sv)/max_abs)*42
        if sv >= 0:
            bar=f'<div style="position:absolute;left:50%;width:{bp}%;height:20px;border-radius:4px;background:linear-gradient(90deg,#ef4444,#dc2626);"></div>'
            lbl=f'<div style="position:absolute;left:calc(50% + {bp}% + 6px);font-size:0.72rem;font-weight:600;color:#64748b;font-family:monospace;white-space:nowrap;">+{sv:.4f}</div>'
        else:
            bar=f'<div style="position:absolute;right:50%;width:{bp}%;height:20px;border-radius:4px;background:linear-gradient(90deg,#2563eb,#3b82f6);"></div>'
            lbl=f'<div style="position:absolute;right:calc(50% + {bp}% + 6px);font-size:0.72rem;font-weight:600;color:#64748b;font-family:monospace;white-space:nowrap;">{sv:.4f}</div>'
        rows += f'<div style="display:flex;align-items:center;margin-bottom:4px;font-size:0.8rem;"><div style="width:120px;min-width:120px;text-align:right;padding-right:12px;font-weight:600;color:#0f172a;font-size:0.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{fn}">{fn}</div><div style="flex:1;position:relative;height:26px;display:flex;align-items:center;"><div style="position:absolute;left:50%;width:2px;height:100%;background:#e2e8f0;z-index:0;"></div>{bar}{lbl}</div><div style="width:75px;min-width:75px;text-align:right;font-size:0.72rem;color:#94a3b8;font-family:monospace;padding-left:8px;">{fv:.4f}</div></div>'

    full_html = f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"><style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:'Inter',-apple-system,sans-serif;background:#fff;padding:24px;}}</style></head><body>
    <div style="margin-bottom:6px;"><div style="font-size:1.05rem;font-weight:700;color:#0f172a;">🔍 SHAP Feature Importance — What Drove This Decision?</div><div style="font-size:0.78rem;color:#475569;margin-top:4px;margin-bottom:12px;">Each bar shows how much a feature pushed the fraud probability up or down. Longer bars = stronger influence.</div></div>
    <div style="display:flex;gap:24px;margin-bottom:16px;font-size:0.76rem;color:#475569;"><div style="display:flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#dc2626;display:inline-block;"></span>Increases Risk ({n_inc} features)</div><div style="display:flex;align-items:center;gap:6px;"><span style="width:12px;height:12px;border-radius:3px;background:#2563eb;display:inline-block;"></span>Decreases Risk ({n_dec} features)</div></div>
    <div style="display:flex;align-items:center;margin-bottom:8px;font-size:0.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;"><div style="width:120px;min-width:120px;text-align:right;padding-right:12px;">Feature</div><div style="flex:1;display:flex;justify-content:space-between;padding:0 10%;"><span>← Decreases Risk</span><span>Increases Risk →</span></div><div style="width:75px;min-width:75px;text-align:right;padding-left:8px;">Value</div></div>
    {rows}</body></html>"""
    components.html(full_html, height=110 + len(shap_explanations)*32, scrolling=False)


def render_lime_chart(lime_explanations):
    """Render LIME explanation as a styled rule-based table via components.html."""
    if not lime_explanations:
        st.markdown(
            '<div class="info-panel">'
            "⚠️ LIME explanation is empty for this transaction. This can happen when "
            "the model is very confident — LIME's local linear approximation finds "
            "no significant contributing rules. The SHAP explanation above provides "
            "the full feature-level breakdown."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rows = ""
    for item in lime_explanations:
        rule = item.get("feature_rule", "?")
        weight = item.get("weight", 0)
        direction = item.get("direction", "")

        if weight > 0:
            color = "#dc2626"
            bg = "#fef2f2"
            icon = "🔴"
            dir_label = "Increases Risk"
        else:
            color = "#2563eb"
            bg = "#eff6ff"
            icon = "🔵"
            dir_label = "Decreases Risk"

        rows += f"""
        <div style="display:flex;align-items:center;padding:10px 14px;margin-bottom:4px;background:{bg};border-radius:8px;border:1px solid {color}15;">
            <div style="font-size:0.9rem;margin-right:10px;">{icon}</div>
            <div style="flex:1;">
                <div style="font-size:0.82rem;font-weight:600;color:#0f172a;font-family:'SF Mono','Fira Code',monospace;">{rule}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">{dir_label}</div>
            </div>
            <div style="font-size:0.85rem;font-weight:700;color:{color};font-family:monospace;min-width:80px;text-align:right;">
                {"+" if weight > 0 else ""}{weight:.4f}
            </div>
        </div>
        """

    full_html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{font-family:'Inter',-apple-system,sans-serif;background:#fff;padding:20px;}}</style>
    </head><body>
    <div style="margin-bottom:12px;">
        <div style="font-size:1.05rem;font-weight:700;color:#0f172a;">🧪 LIME Explanation — Rule-Based Interpretation</div>
        <div style="font-size:0.78rem;color:#475569;margin-top:4px;margin-bottom:16px;">
            LIME builds a local linear model around this specific prediction, expressing feature contributions as human-readable rules.
            Unlike SHAP (which uses game theory), LIME approximates the model's behavior locally.
        </div>
    </div>
    {rows}
    <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;font-size:0.72rem;color:#64748b;">
        💡 <strong>SHAP vs LIME:</strong> SHAP provides exact Shapley values (mathematically guaranteed). LIME provides an approximate local linear explanation. Both are complementary — SHAP for precision, LIME for intuitive rule-based insights.
    </div>
    </body></html>"""

    chart_height = 120 + len(lime_explanations) * 58
    components.html(full_html, height=chart_height, scrolling=False)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "transaction_features" not in st.session_state: st.session_state.transaction_features = None
if "prediction_result" not in st.session_state: st.session_state.prediction_result = None
if "lime_result" not in st.session_state: st.session_state.lime_result = None
if "input_mode" not in st.session_state: st.session_state.input_mode = "welcome"
if "feedback_submitted" not in st.session_state: st.session_state.feedback_submitted = None

# ---------------------------------------------------------------------------
# API CHECK
# ---------------------------------------------------------------------------
health = check_api_health()
api_online = health is not None

# ---------------------------------------------------------------------------
# TOP NAVBAR
# ---------------------------------------------------------------------------
if api_online:
    dot_class, status_text = "online", "System Operational"
    uptime = time.strftime("%H:%M:%S", time.gmtime(health.get("uptime_seconds", 0)))
else:
    dot_class, status_text, uptime = "offline", "API Offline", "--:--:--"

st.markdown(f'<div class="top-navbar"><div class="navbar-brand"><span class="navbar-logo">🛡️</span><div><div class="navbar-title">{DASHBOARD_TITLE}</div><div class="navbar-subtitle">Enterprise Fraud Intelligence Platform</div></div></div><div class="navbar-status"><div class="navbar-status-item"><span class="status-dot {dot_class}"></span> {status_text}</div><div class="navbar-status-item">⏱️ Uptime: {uptime}</div><div class="navbar-status-item">🔗 API: {API_URL}</div></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Panel")
    st.markdown("---")
    st.markdown("#### Component Status")
    if api_online:
        comps = {"Model":health.get("model_loaded"),"SHAP":health.get("shap_loaded"),"LIME":health.get("lime_loaded"),"Scaler":health.get("scaler_loaded")}
        ch='<div class="component-grid">'
        for n,l in comps.items(): ch+=f'<div class="component-chip">{"✅" if l else "❌"} {n}</div>'
        ch+="</div>"
        st.markdown(ch, unsafe_allow_html=True)
        st.caption(f"Features: {health.get('feature_count','?')} · Version: v1 · Threshold: 0.585")
    else:
        st.error("API is not running"); st.code("python -m src.api.inference_api", language="bash")

    st.markdown("---")
    st.markdown("#### 🚀 Quick Actions")
    if api_online:
        if st.button("📋 New Analysis", use_container_width=True, type="primary"):
            s = get_sample_transaction()
            if s:
                st.session_state.transaction_features=s; st.session_state.prediction_result=None
                st.session_state.lime_result=None; st.session_state.feedback_submitted=None; st.session_state.input_mode="input"; st.rerun()
        if st.button("🔄 Reset Dashboard", use_container_width=True):
            st.session_state.transaction_features=None; st.session_state.prediction_result=None
            st.session_state.lime_result=None; st.session_state.feedback_submitted=None; st.session_state.input_mode="welcome"; st.rerun()
        if st.session_state.input_mode == "results":
            if st.button("✏️ Edit & Re-analyze", use_container_width=True):
                st.session_state.prediction_result=None; st.session_state.lime_result=None
                st.session_state.feedback_submitted=None; st.session_state.input_mode="input"; st.rerun()

    st.markdown("---")
    st.markdown("#### 📋 Risk Levels")
    for lvl,lbl in [("low","LOW"),("medium","MEDIUM"),("high","HIGH"),("critical","CRITICAL")]:
        th={"low":"< 20%","medium":"20–49%","high":"50–79%","critical":"≥ 80%"}
        st.markdown(f'<span class="risk-badge {lvl}">{lbl}</span> &nbsp; {th[lvl]}', unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Dashboard v1.1 · Port {config['dashboard']['port']}")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_analysis, tab_network, tab_performance = st.tabs(["🔍  Transaction Analysis","🕸️  Fraud Ring Network","📊  Model Performance"])

# ---------------------------------------------------------------------------
# TAB: Transaction Analysis
# ---------------------------------------------------------------------------
with tab_analysis:
    if not api_online:
        st.markdown('<div class="info-panel">⚠️ <strong>Cannot analyze transactions</strong> — API offline.</div>', unsafe_allow_html=True)

    elif st.session_state.input_mode == "welcome":
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown('<div class="kpi-card blue"><div class="kpi-label">API Status</div><div class="kpi-value" style="color:#059669;">Online</div><div class="kpi-sub">All systems operational</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Input Features</div><div class="kpi-value">{health.get("feature_count","?")}</div><div class="kpi-sub">PCA + engineered features</div></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="kpi-card navy"><div class="kpi-label">Active Model</div><div class="kpi-value">XGBoost</div><div class="kpi-sub">Optimized for AUPRC</div></div>', unsafe_allow_html=True)
        with c4: st.markdown('<div class="kpi-card amber"><div class="kpi-label">Decision Threshold</div><div class="kpi-value">0.585</div><div class="kpi-sub">Optimal precision-recall</div></div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="info-panel">👈 <strong>Get started:</strong> Click <strong>"New Analysis"</strong> in the sidebar to load a transaction template, adjust feature values, then run the fraud analysis.</div>', unsafe_allow_html=True)

    elif st.session_state.input_mode == "input":
        features = st.session_state.transaction_features
        if features is None: st.session_state.input_mode="welcome"; st.rerun()

        input_col, preview_col = st.columns([3, 2])
        with input_col:
            st.markdown('<div class="input-card-header">💰 Transaction Details</div>', unsafe_allow_html=True)
            td1,td2 = st.columns(2)
            with td1: features["Amount"]=st.number_input("Amount ($)",value=float(features.get("Amount",0.0)),min_value=0.0,max_value=50000.0,step=10.0,format="%.2f",key="in_Amount")
            with td2: features["Time"]=st.number_input("Time (sec)",value=float(features.get("Time",0.0)),min_value=0.0,max_value=200000.0,step=100.0,format="%.0f",key="in_Time")
            st.markdown("")
            st.markdown('<div class="input-card-header">🔧 Engineered Features</div>', unsafe_allow_html=True)
            e1,e2,e3,e4=st.columns(4)
            with e1:
                features["amount_log"]=st.number_input("Amount (log)",value=float(features.get("amount_log",0.0)),step=0.1,format="%.4f",key="in_alog")
                features["is_night"]=st.number_input("Is Night",value=float(features.get("is_night",0.0)),min_value=0.0,max_value=1.0,step=1.0,format="%.0f",key="in_night")
            with e2:
                features["hour_sin"]=st.number_input("Hour (sin)",value=float(features.get("hour_sin",0.0)),min_value=-1.0,max_value=1.0,step=0.1,format="%.4f",key="in_hsin")
                features["freq_1h"]=st.number_input("Freq (1h)",value=float(features.get("freq_1h",0.0)),min_value=0.0,step=1.0,format="%.0f",key="in_f1h")
            with e3:
                features["hour_cos"]=st.number_input("Hour (cos)",value=float(features.get("hour_cos",0.0)),min_value=-1.0,max_value=1.0,step=0.1,format="%.4f",key="in_hcos")
                features["freq_24h"]=st.number_input("Freq (24h)",value=float(features.get("freq_24h",0.0)),min_value=0.0,step=1.0,format="%.0f",key="in_f24h")
            with e4:
                features["amount_zscore"]=st.number_input("Z-score",value=float(features.get("amount_zscore",0.0)),step=0.1,format="%.4f",key="in_az")
                features["time_since_last"]=st.number_input("Since Last (s)",value=float(features.get("time_since_last",0.0)),min_value=0.0,step=10.0,format="%.0f",key="in_tsl")
            st.markdown("")
            with st.expander("🔬 PCA Features (V1–V28)", expanded=False):
                st.caption("Anonymized PCA components.")
                pcols=st.columns(4)
                for i in range(1,29):
                    fn=f"V{i}"
                    with pcols[(i-1)%4]: features[fn]=st.number_input(fn,value=float(features.get(fn,0.0)),step=0.1,format="%.4f",key=f"in_{fn}")
            st.markdown("")
            ac1,ac2,ac3=st.columns([1,2,1])
            with ac2:
                if st.button("🔍 Analyze Transaction",use_container_width=True,type="primary"):
                    with st.spinner("Running fraud analysis..."):
                        result=predict_transaction(features)
                        if result:
                            st.session_state.prediction_result=result
                            st.session_state.lime_result=None
                            st.session_state.feedback_submitted=None
                            st.session_state.input_mode="results"; st.rerun()
                        else: st.error("Prediction failed.")

        with preview_col:
            nz=sum(1 for v in features.values() if v!=0.0)
            st.markdown(f'<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;font-weight:600;">📊 {nz}/{len(features)} features non-zero</div>', unsafe_allow_html=True)
            st.markdown(build_preview_html(features), unsafe_allow_html=True)
        st.session_state.transaction_features = features

    elif st.session_state.input_mode == "results":
        result = st.session_state.prediction_result
        if result is None: st.session_state.input_mode="welcome"; st.rerun()

        prob=result.get("fraud_probability",0); risk_level=result.get("risk_level","UNKNOWN")
        is_flagged=result.get("is_flagged",False); prediction_label=result.get("prediction_label","UNKNOWN")
        threshold=result.get("threshold_used",0.585); inference_ms=result.get("inference_time_ms",0)
        plain_english=result.get("plain_english_summary",""); shap_data=result.get("shap_explanation",[])

        rc={"LOW":{"bg":"#d1fae5","text":"#065f46","g":"#059669"},"MEDIUM":{"bg":"#fef3c7","text":"#92400e","g":"#d97706"},"HIGH":{"bg":"#fed7aa","text":"#9a3412","g":"#ea580c"},"CRITICAL":{"bg":"#fecaca","text":"#991b1b","g":"#dc2626"}}
        c=rc.get(risk_level,rc["LOW"])

        # KPI Row
        k1,k2,k3,k4=st.columns(4)
        with k1: st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Fraud Probability</div><div class="kpi-value" style="color:{c["g"]};">{prob:.1%}</div><div class="kpi-sub">Model confidence score</div></div>', unsafe_allow_html=True)
        with k2:
            bc=risk_level.lower()
            st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Risk Level</div><div class="kpi-value"><span class="risk-badge {bc}" style="font-size:1.1rem;padding:0.3rem 1rem;">{risk_level}</span></div><div class="kpi-sub">Classification tier</div></div>', unsafe_allow_html=True)
        with k3:
            fi="🚨" if is_flagged else "✅"; ft="FLAGGED" if is_flagged else "CLEAR"
            fc=c["g"] if is_flagged else "#059669"; ca="amber" if is_flagged else "navy"
            st.markdown(f'<div class="kpi-card {ca}"><div class="kpi-label">Decision</div><div class="kpi-value" style="color:{fc};">{fi} {ft}</div><div class="kpi-sub">{prediction_label}</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="kpi-card navy"><div class="kpi-label">Inference Time</div><div class="kpi-value">{inference_ms:.0f}ms</div><div class="kpi-sub">Threshold: {threshold:.3f}</div></div>', unsafe_allow_html=True)

        # Gauge + Summary
        st.markdown("")
        gauge_col, summary_col = st.columns([2, 3])
        with gauge_col:
            svg=build_gauge_svg(prob,risk_level)
            st.markdown(f'<div class="gauge-wrapper"><div class="gauge-title">Fraud Risk Assessment</div><div class="gauge-svg-container">{svg}</div><div class="gauge-value-large" style="color:{c["g"]};">{prob:.1%}</div><div class="gauge-risk-label" style="background:{c["bg"]};color:{c["text"]};">{risk_level} RISK</div><div class="gauge-decision-row"><div class="gauge-decision-item">{"🚨" if is_flagged else "✅"} {prediction_label}</div><div class="gauge-decision-item">⚡ {inference_ms:.0f}ms</div></div></div>', unsafe_allow_html=True)
        with summary_col:
            st.markdown('<div class="section-header">📋 Analysis Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-panel">{plain_english}</div>', unsafe_allow_html=True)
            st.caption(f"Transaction: {result.get('transaction_id','N/A')} · Model: v1 · Analyzed: {result.get('timestamp','N/A')}")

        # SHAP Chart
        st.markdown("")
        render_shap_chart(shap_data)

        # ============================================================
        # STEP 5: LIME EXPLANATION PANEL
        # ============================================================
        st.markdown("")
        with st.expander("🧪 LIME Explanation — Alternative Interpretability", expanded=False):
            st.markdown(
                "LIME (Local Interpretable Model-agnostic Explanations) provides "
                "a complementary view to SHAP by building a local linear approximation "
                "around this specific prediction. Click below to generate."
            )

            if st.session_state.lime_result is not None:
                lime_data = st.session_state.lime_result
                lime_explanations = lime_data.get("lime_explanation", [])
                lime_prob = lime_data.get("fraud_probability", prob)
                lime_risk = lime_data.get("risk_level", risk_level)
                lime_summary = lime_data.get("plain_english_summary", "")

                lk1, lk2, lk3 = st.columns(3)
                with lk1:
                    st.metric("LIME Probability", f"{lime_prob:.1%}")
                with lk2:
                    st.metric("LIME Risk Level", lime_risk)
                with lk3:
                    st.metric("Rules Found", len(lime_explanations))

                if lime_summary:
                    st.markdown(f'<div class="info-panel">{lime_summary}</div>', unsafe_allow_html=True)

                render_lime_chart(lime_explanations)

            else:
                if st.button("🧪 Generate LIME Explanation", use_container_width=True):
                    features = st.session_state.transaction_features
                    if features:
                        with st.spinner("Generating LIME explanation (this may take a few seconds)..."):
                            lime_res = predict_lime(features)
                            if lime_res:
                                st.session_state.lime_result = lime_res
                                st.rerun()
                            else:
                                st.error("LIME prediction failed or returned an error. Check API logs.")
                    else:
                        st.warning("No transaction features available.")

        # ============================================================
        # PHASE 10: INVESTIGATOR FEEDBACK PANEL
        # ============================================================
        st.markdown("")
        st.markdown('<div class="section-header">📝 Investigator Feedback</div>', unsafe_allow_html=True)

        if st.session_state.feedback_submitted:
            fb = st.session_state.feedback_submitted
            retrain = fb.get("retrain_status", {})
            correction_label = fb.get("message", "Feedback recorded.")
            remaining = retrain.get("remaining", "?")
            total_corr = retrain.get("total_corrections", 0)
            threshold_val = retrain.get("threshold", 100)

            if retrain.get("retrain_recommended", False):
                st.success(f"✅ {correction_label} Retrain threshold REACHED ({total_corr}/{threshold_val} corrections). Consider retraining the model.")
            else:
                st.success(f"✅ {correction_label} ({total_corr}/{threshold_val} corrections — {remaining} more until retrain)")
        else:
            st.markdown(
                '<div class="info-panel">'
                'As an investigator, you can override the model\'s decision. '
                'Your feedback is stored and used to improve future model versions.'
                '</div>',
                unsafe_allow_html=True,
            )

            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                feedback_notes = st.text_area(
                    "Investigator Notes (optional)",
                    placeholder="E.g., Customer confirmed this purchase. / Matches known fraud pattern.",
                    height=80,
                    key="fb_notes",
                )
            with fb_col2:
                st.markdown("")
                st.markdown("")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Confirmed Fraud", use_container_width=True, type="primary"):
                        features = st.session_state.transaction_features
                        if features and result:
                            with st.spinner("Submitting feedback..."):
                                fb_result = submit_feedback(features, result, "confirmed_fraud", feedback_notes)
                                if fb_result and fb_result.get("success"):
                                    st.session_state.feedback_submitted = fb_result
                                    st.rerun()
                                else:
                                    st.error("Failed to submit feedback. Check API logs.")
                with btn_col2:
                    if st.button("❌ False Positive", use_container_width=True):
                        features = st.session_state.transaction_features
                        if features and result:
                            with st.spinner("Submitting feedback..."):
                                fb_result = submit_feedback(features, result, "false_positive", feedback_notes)
                                if fb_result and fb_result.get("success"):
                                    st.session_state.feedback_submitted = fb_result
                                    st.rerun()
                                else:
                                    st.error("Failed to submit feedback. Check API logs.")

        # Feedback History (collapsible)
        with st.expander("📜 Recent Feedback History", expanded=False):
            history = get_feedback_history_from_api(limit=10)
            if history and history.get("records"):
                records = history["records"]
                st.caption(f"Showing {len(records)} of {history.get('total', 0)} total corrections")
                for rec in records:
                    corr = rec.get("correction_type", "?")
                    corr_icon = "🔴 Confirmed Fraud" if corr == "confirmed_fraud" else "🟢 False Positive"
                    orig_prob = rec.get("original_probability", 0)
                    orig_risk = rec.get("original_risk_level", "?")
                    notes = rec.get("investigator_notes", "")
                    ts = rec.get("created_at", "?")
                    txn_short = rec.get("transaction_id", "?")[:12] + "..."

                    notes_display = f" — \"{notes}\"" if notes else ""
                    st.markdown(
                        f"**{corr_icon}** | Txn: `{txn_short}` | "
                        f"Original: {orig_prob:.1%} ({orig_risk}) | "
                        f"{ts[:19]}{notes_display}"
                    )
            else:
                st.caption("No feedback recorded yet. Submit your first correction above.")

            # Stats
            stats_data = get_feedback_stats_from_api()
            if stats_data and stats_data.get("stats", {}).get("total", 0) > 0:
                s = stats_data["stats"]
                r = stats_data.get("retrain_status", {})
                st.markdown("---")
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: st.metric("Total Corrections", s.get("total", 0))
                with sc2: st.metric("Confirmed Fraud", s.get("confirmed_fraud", 0))
                with sc3: st.metric("False Positives", s.get("false_positive", 0))
                with sc4: st.metric("Until Retrain", r.get("remaining", "?"))

# ---------------------------------------------------------------------------
# TAB: Fraud Ring Network
# ---------------------------------------------------------------------------
with tab_network:
    st.markdown('<div class="section-header">🕸️ Fraud Ring Network Visualization</div>', unsafe_allow_html=True)

    graph_summary = load_graph_summary()
    rings_data = load_fraud_rings()
    edge_data = load_edge_list()

    if graph_summary is None:
        st.markdown('<div class="info-panel">⚠️ Graph data not found. Run Phase 4 first: <code>python -m src.graph_analytics.graph_builder</code></div>', unsafe_allow_html=True)
    else:
        gs = graph_summary.get("graph_stats", {})
        rs = graph_summary.get("ring_stats", {})

        nk1, nk2, nk3, nk4, nk5 = st.columns(5)
        with nk1:
            st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Total Nodes</div><div class="kpi-value">{gs.get("total_nodes",0)}</div><div class="kpi-sub">Fraud transactions</div></div>', unsafe_allow_html=True)
        with nk2:
            st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Total Edges</div><div class="kpi-value">{gs.get("total_edges",0)}</div><div class="kpi-sub">Connections found</div></div>', unsafe_allow_html=True)
        with nk3:
            st.markdown(f'<div class="kpi-card amber"><div class="kpi-label">Fraud Rings</div><div class="kpi-value">{rs.get("total_rings_detected",0)}</div><div class="kpi-sub">Coordinated clusters</div></div>', unsafe_allow_html=True)
        with nk4:
            st.markdown(f'<div class="kpi-card navy"><div class="kpi-label">Isolated Nodes</div><div class="kpi-value">{rs.get("isolated_nodes",0)}</div><div class="kpi-sub">One-off fraud cases</div></div>', unsafe_allow_html=True)
        with nk5:
            density_pct = gs.get("density", 0) * 100
            st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Graph Density</div><div class="kpi-value">{density_pct:.1f}%</div><div class="kpi-sub">Network connectivity</div></div>', unsafe_allow_html=True)

        st.markdown("")

        strategy = graph_summary.get("strategy", "unknown")
        params = graph_summary.get("parameters", {})
        st.markdown(
            f'<div class="info-panel">'
            f'<strong>Detection Strategy:</strong> {strategy.replace("_"," ").title()} — '
            f'Similarity threshold: {params.get("similarity_threshold", "?")} | '
            f'Time window: {params.get("time_window_seconds", "?")}s | '
            f'Min ring size: {params.get("min_ring_size", "?")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if edge_data:
            st.markdown('<div class="section-header">🌐 Interactive Network Graph</div>', unsafe_allow_html=True)
            st.caption("Drag nodes to explore. Hover for details. Scroll to zoom. Clusters = fraud rings.")
            viz_html = build_network_viz_html(edge_data, rings_data)
            if viz_html:
                components.html(viz_html, height=620, scrolling=False)
        else:
            st.markdown('<div class="info-panel">Edge list not found at graphs/fraud_edges.csv. Run Phase 4 first.</div>', unsafe_allow_html=True)

        if rings_data:
            st.markdown('<div class="section-header">🔎 Detected Fraud Rings</div>', unsafe_allow_html=True)
            render_ring_details(rings_data)

# ---------------------------------------------------------------------------
# TAB: Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.markdown('<div class="section-header">📊 Model Performance &amp; Comparison</div>', unsafe_allow_html=True)

    metrics_data = load_metrics()
    comparison_data = load_comparison()

    if metrics_data is None:
        st.markdown('<div class="info-panel">⚠️ Metrics data not found. Run Phase 5 first: <code>python -m src.models.model_training</code></div>', unsafe_allow_html=True)
    else:
        if comparison_data:
            winner = comparison_data.get("winner", "?")
            winning_auprc = comparison_data.get("winning_auprc", 0)
            margin = comparison_data.get("margin", 0)
            st.markdown(
                f'<div class="info-panel">'
                f'🏆 <strong>{winner} is the deployed model</strong> with an AUPRC of '
                f'<strong>{winning_auprc:.6f}</strong> (margin: +{margin:.6f} over the runner-up). '
                f'AUPRC is the primary evaluation metric — standard accuracy is forbidden for '
                f'highly imbalanced financial data.'
                f'</div>',
                unsafe_allow_html=True,
            )

        xgb = metrics_data.get("xgboost", {})
        lgb = metrics_data.get("lightgbm", {})

        pk1, pk2, pk3, pk4 = st.columns(4)
        with pk1:
            st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">XGBoost AUPRC</div><div class="kpi-value" style="color:#059669;">{xgb.get("auprc",0):.4f}</div><div class="kpi-sub">Primary metric (winner)</div></div>', unsafe_allow_html=True)
        with pk2:
            st.markdown(f'<div class="kpi-card green"><div class="kpi-label">LightGBM AUPRC</div><div class="kpi-value">{lgb.get("auprc",0):.4f}</div><div class="kpi-sub">Runner-up</div></div>', unsafe_allow_html=True)
        with pk3:
            xgb_prec = xgb.get("default_threshold", {}).get("precision", 0)
            st.markdown(f'<div class="kpi-card amber"><div class="kpi-label">XGBoost Precision</div><div class="kpi-value">{xgb_prec:.1%}</div><div class="kpi-sub">At default threshold</div></div>', unsafe_allow_html=True)
        with pk4:
            xgb_rec = xgb.get("default_threshold", {}).get("recall", 0)
            st.markdown(f'<div class="kpi-card navy"><div class="kpi-label">XGBoost Recall</div><div class="kpi-value">{xgb_rec:.1%}</div><div class="kpi-sub">At default threshold</div></div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">⚔️ Head-to-Head: XGBoost vs LightGBM</div>', unsafe_allow_html=True)

        if comparison_data:
            render_comparison_table(comparison_data)
        else:
            st.markdown('<div class="info-panel">Comparison data not found at models/model_comparison.json</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

        insights_col1, insights_col2 = st.columns(2)
        with insights_col1:
            st.markdown(
                '<div class="info-panel">'
                '<strong>Why AUPRC?</strong> In fraud detection, only ~0.17% of transactions '
                'are fraudulent. Standard accuracy would be 99.83% by simply predicting "not fraud" '
                'every time. AUPRC correctly measures performance on the rare positive class, '
                'making it the only honest metric for imbalanced financial data.'
                '</div>',
                unsafe_allow_html=True,
            )
        with insights_col2:
            st.markdown(
                '<div class="info-panel">'
                '<strong>Why XGBoost Won:</strong> XGBoost achieved a significantly higher AUPRC '
                'and maintains strong precision at a reasonable threshold (0.585). LightGBM requires '
                'an extremely high threshold (0.906) to achieve acceptable precision, making it less '
                'practical for real-time deployment.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown('<div class="section-header">🛡️ Adversarial Stress Test Results</div>', unsafe_allow_html=True)

        stress_data = None
        try:
            stress_path = resolve_path(config["stress_test"]["results_path"])
            if stress_path.exists():
                with open(stress_path, "r") as f:
                    stress_data = json.load(f)
        except Exception:
            pass

        if stress_data:
            summary = stress_data.get("summary", {})
            total = summary.get("total_tests", 0)
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)

            sk1, sk2, sk3 = st.columns(3)
            with sk1:
                st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Total Tests</div><div class="kpi-value">{total}</div><div class="kpi-sub">Adversarial scenarios</div></div>', unsafe_allow_html=True)
            with sk2:
                st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Passed</div><div class="kpi-value" style="color:#059669;">{passed}</div><div class="kpi-sub">Model robust</div></div>', unsafe_allow_html=True)
            with sk3:
                st.markdown(f'<div class="kpi-card {"amber" if failed > 0 else "navy"}""><div class="kpi-label">Failed</div><div class="kpi-value" style="color:{"#d97706" if failed > 0 else "#059669"};">{failed}</div><div class="kpi-sub">Vulnerabilities found</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-panel">Stress test results not found. Run Phase 6 first.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown('<div class="app-footer">Explainable Fraud Detection System · Phase 10: Human-in-the-Loop & CI/CD · Built with transparency, auditability, and FCA compliance in mind</div>', unsafe_allow_html=True)
logger.info("Dashboard page loaded successfully")
