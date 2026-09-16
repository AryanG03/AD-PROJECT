"""
dashboard.py
────────────
Neuro-CX Interactive Streamlit Dashboard

Panels:
  1. Sidebar  — model + customer selector
  2. Sequence Stepper — step through interactions one at a time
  3. Recommendations — live ranked list updating after each step
  4. Hidden-State Visualizer — heatmap/line of state vector over time
  5. Metrics Panel — comparison chart + table from evaluation output

Run with:
  streamlit run src/app/dashboard.py
  (or: py run_app.py)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model.neuro_cx_model import NeuroCXModel
from model.trainer import load_checkpoint

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-CX | Adaptive CX Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0;
  }
  .hero-sub {
    color: #888;
    font-size: 0.95rem;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }
  .action-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .badge-purchase    { background: #2d6a4f; color: #74c69d; }
  .badge-view        { background: #23395d; color: #74b9ff; }
  .badge-add_to_cart { background: #4a2c6d; color: #c77dff; }
  .badge-review      { background: #6b3a1f; color: #fca46d; }
  .badge-bounce      { background: #4a1942; color: #f08080; }

  .rec-item {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    transition: all 0.2s;
  }
  .rec-rank { color: #888; font-size: 0.8rem; min-width: 28px; }
  .rec-name { color: #e0e0e0; font-size: 0.88rem; flex: 1; }
  .rec-score { color: #667eea; font-size: 0.8rem; font-weight: 600; }

  .step-info {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

  div[data-testid="stSidebar"] { background: #0d0d1a; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "purchase":    "#74c69d",
    "view":        "#74b9ff",
    "add_to_cart": "#c77dff",
    "review":      "#fca46d",
    "bounce":      "#f08080",
    "unknown":     "#aaa",
}

ACTION_ICONS = {
    "purchase":    "🛒",
    "view":        "👁️",
    "add_to_cart": "🛍️",
    "review":      "⭐",
    "bounce":      "↩️",
    "unknown":     "•",
}

ITEM_CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Beauty", "Toys", "Food & Beverages", "Automotive", "Gardening"
]
ITEMS_PER_CAT = 20


def item_id_to_name(item_id: int) -> str:
    """Convert encoded item ID back to a human-readable label."""
    cat_idx  = item_id // ITEMS_PER_CAT
    item_num = item_id % ITEMS_PER_CAT + 1
    cat = ITEM_CATEGORIES[cat_idx] if cat_idx < len(ITEM_CATEGORIES) else "Misc"
    return f"{cat} #{item_num:02d}"


# ── Session state helpers ─────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "step":           0,
        "hidden_history": [],
        "rec_history":    [],
        "action_history": [],
        "item_history":   [],
        "current_model":  None,
        "model_loaded":   False,
        "sequences":      None,
        "customer_idx":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_stepper():
    st.session_state.step           = 0
    st.session_state.hidden_history = []
    st.session_state.rec_history    = []
    st.session_state.action_history = []
    st.session_state.item_history   = []


# ── Data / model loading (cached) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_data = load_checkpoint(checkpoint_path, device)
    model.eval()
    return model, ckpt_data, device


@st.cache_data(show_spinner="Loading interaction sequences…")
def load_sequences(interactions_path: str):
    if not os.path.exists(interactions_path):
        return None
    df = pd.read_parquet(interactions_path)
    return df


@st.cache_data(show_spinner="Loading encoder metadata…")
def load_encoders(enc_path: str):
    if not os.path.exists(enc_path):
        return None
    with open(enc_path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading evaluation results…")
def load_comparison(json_path: str):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


# ── Inference helpers ─────────────────────────────────────────────────────────

def build_tensors_for_step(
    customer_df: pd.DataFrame,
    step: int,
    enc_meta: dict,
    max_seq_len: int = 50,
    device: torch.device = None,
):
    """
    Build input tensors for steps 0..step (inclusive) from a customer's
    interaction history, left-padded to max_seq_len.
    """
    if device is None:
        device = torch.device("cpu")

    action_weights = enc_meta.get("action_weights", {})
    action_classes = enc_meta.get("action_classes", [])
    item_map       = {int(k): v for k, v in enc_meta.get("item_map", {}).items()}
    dwell_max      = enc_meta.get("dwell_max", 600.0)

    rows = customer_df.iloc[:step + 1]

    def encode_action(a):
        a = str(a).strip()
        if a in action_classes:
            return action_classes.index(a)
        return 0

    def encode_item(i):
        return item_map.get(int(i), 0)

    actions  = [encode_action(r["action_type"]) for _, r in rows.iterrows()]
    items    = [encode_item(r["item_id"])        for _, r in rows.iterrows()]
    dwells   = [min(float(r["dwell_time"]) / dwell_max, 1.0) for _, r in rows.iterrows()]
    weights  = [float(action_weights.get(str(r["action_type"]).strip(), 1.0)) for _, r in rows.iterrows()]

    actual_len = len(actions)
    if actual_len > max_seq_len:
        actions = actions[-max_seq_len:]
        items   = items[-max_seq_len:]
        dwells  = dwells[-max_seq_len:]
        weights = weights[-max_seq_len:]
        actual_len = max_seq_len

    pad = max_seq_len - actual_len
    actions = [0] * pad + actions
    items   = [0] * pad + items
    dwells  = [0.0] * pad + dwells
    weights = [0.0] * pad + weights

    return (
        torch.tensor([actions],  dtype=torch.long,    device=device),
        torch.tensor([items],    dtype=torch.long,    device=device),
        torch.tensor([dwells],   dtype=torch.float32, device=device),
        torch.tensor([weights],  dtype=torch.float32, device=device),
        actual_len,
    )


@torch.no_grad()
def get_recommendations(
    model, action_seq, item_seq, dwell_seq, weight_seq, top_k=10
):
    """Run forward pass and return (ranked_items, hidden_state, all_hiddens)."""
    is_neuro_cx = isinstance(model, NeuroCXModel)
    if is_neuro_cx:
        logits, hidden, all_hiddens = model(
            item_seq, action_seq, dwell_seq, weight_seq,
            return_all_hidden=True
        )
    else:
        logits, hidden = model(item_seq, action_seq, dwell_seq)
        all_hiddens = [hidden[-1].squeeze(0)]

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top_k_idx = np.argsort(probs)[::-1][:top_k].tolist()
    top_k_scores = probs[top_k_idx].tolist()

    last_hidden = all_hiddens[-1].squeeze(0).cpu().numpy() if all_hiddens else None

    return top_k_idx, top_k_scores, last_hidden, all_hiddens


# ── UI Panels ─────────────────────────────────────────────────────────────────

def render_header():
    st.markdown('<p class="hero-title">🧠 Neuro-CX</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Adaptive Customer Experience — GRU with Reinforcement & Forgetting Mechanisms</p>',
        unsafe_allow_html=True
    )


def render_sidebar(enc_meta):
    st.sidebar.markdown("## ⚙️ Configuration")

    # Model selector
    st.sidebar.markdown("### Model")
    model_choice = st.sidebar.radio(
        "Select model", ["Neuro-CX (Reinforcement + Decay)", "GRU Baseline"],
        key="model_choice_radio"
    )
    is_neuro_cx = "Neuro-CX" in model_choice

    ckpt_dir = os.path.join(ROOT, "checkpoints")
    ckpt_file = "neuro_cx_best.pt" if is_neuro_cx else "gru_baseline_best.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    # Top-k slider
    top_k = st.sidebar.slider("Recommendations to show (k)", 5, 20, 10, key="top_k_slider")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Device")
    device_str = "🟢 CUDA (GPU)" if torch.cuda.is_available() else "🟡 CPU"
    st.sidebar.info(device_str)
    if torch.cuda.is_available():
        st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")

    return ckpt_path, is_neuro_cx, top_k


def render_recommendations(top_k_idx, top_k_scores, n_items_total):
    st.markdown("### 🎯 Current Recommendations")
    for rank, (item_id, score) in enumerate(zip(top_k_idx, top_k_scores), 1):
        name = item_id_to_name(item_id)
        pct  = score * 100
        int(pct / max(top_k_scores) * 100) if max(top_k_scores) > 0 else 0
        st.markdown(
            f'<div class="rec-item">'
            f'<span class="rec-rank">#{rank}</span>'
            f'<span class="rec-name">{name}</span>'
            f'<span class="rec-score">{pct:.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_hidden_state_viz(hidden_history: list, action_history: list):
    """
    Render two charts:
      - Heatmap of hidden-state vector evolution over time
      - Mean absolute activation line chart
    """
    if not hidden_history:
        st.info("Step through the sequence to see the hidden state evolve.")
        return

    st.markdown("### 🧬 Hidden State Evolution")

    # Stack into (T, H) matrix
    H = np.stack([h[:64] for h in hidden_history], axis=0)  # show first 64 dims
    T, D = H.shape
    steps = [f"t{i+1}" for i in range(T)]

    # ── Heatmap ────────────────────────────────────────────────────────────────
    fig_heat = go.Figure(go.Heatmap(
        z=H.T,
        x=steps,
        y=[f"h{i}" for i in range(D)],
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar={"title": "Activation", "thickness": 12, "len": 0.8},
    ))
    # Annotate action type on each step
    for i, act in enumerate(action_history):
        ACTION_COLORS.get(act, "#aaa")
        fig_heat.add_annotation(
            x=steps[i], y=D - 0.5,
            text=ACTION_ICONS.get(act, "•"),
            showarrow=False,
            font={"size": 12},
            yref="y",
        )

    fig_heat.update_layout(
        title={"text": "Hidden State Heatmap (first 64 dims)", "font": {"size": 13}},
        xaxis_title="Time step",
        yaxis_title="Hidden dim",
        height=320,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Mean activation line ────────────────────────────────────────────────
    mean_act = H.mean(axis=1)
    std_act  = H.std(axis=1)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=steps, y=mean_act + std_act,
        fill=None, mode="lines",
        line={"width": 0},
        showlegend=False,
        name="Upper bound",
    ))
    fig_line.add_trace(go.Scatter(
        x=steps, y=mean_act - std_act,
        fill="tonexty", mode="lines",
        line={"width": 0},
        fillcolor="rgba(102,126,234,0.15)",
        showlegend=False,
        name="Std band",
    ))
    fig_line.add_trace(go.Scatter(
        x=steps, y=mean_act,
        mode="lines+markers",
        line={"color": "#667eea", "width": 2.5},
        marker={
            "size": [10 if a == "purchase" else 6 for a in action_history],
            "color": [ACTION_COLORS.get(a, "#aaa") for a in action_history],
            "symbol": ["star" if a == "purchase" else "circle" for a in action_history],
            "line": {"width": 1, "color": "white"},
        },
        name="Mean activation",
    ))

    fig_line.update_layout(
        title={"text": "Mean Hidden Activation (⭐ = purchase event)", "font": {"size": 13}},
        xaxis_title="Time step",
        yaxis_title="Mean |h|",
        height=250,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        showlegend=False,
    )
    st.plotly_chart(fig_line, use_container_width=True)


def compute_customer_metrics(model_baseline, model_neurocx, customer_df, enc_meta, max_seq_len, device):
    n_events = len(customer_df)
    if n_events < 2:
        return None, None

    baseline_logits_all = []
    neurocx_logits_all = []
    targets_all = []

    # Map item_id to encoded item_id
    item_map = {int(k): v for k, v in enc_meta.get("item_map", {}).items()}

    # Evaluate sequentially at each transition point in the sequence
    for s in range(n_events - 1):
        action_seq_t, item_seq_t, dwell_seq_t, weight_seq_t, _ = build_tensors_for_step(
            customer_df, s, enc_meta, max_seq_len, device
        )
        
        # Next item in sequence is the target
        next_event = customer_df.iloc[s + 1]
        target_id = int(next_event["item_id"])
        target_enc = item_map.get(target_id, 0)

        with torch.no_grad():
            b_logits, _ = model_baseline(item_seq_t, action_seq_t, dwell_seq_t)
            n_logits, _ = model_neurocx(item_seq_t, action_seq_t, dwell_seq_t, weight_seq_t)

        baseline_logits_all.append(b_logits.cpu())
        neurocx_logits_all.append(n_logits.cpu())
        targets_all.append(target_enc)

    # Concatenate results
    b_logits = torch.cat(baseline_logits_all, dim=0)  # (T-1, n_items)
    n_logits = torch.cat(neurocx_logits_all, dim=0)  # (T-1, n_items)
    targets = torch.tensor(targets_all, dtype=torch.long)  # (T-1,)

    from evaluation.metrics import compute_all_metrics
    k_values = [5, 10, 20]
    
    b_metrics = compute_all_metrics(b_logits, targets, k_values=k_values)
    n_metrics = compute_all_metrics(n_logits, targets, k_values=k_values)

    return b_metrics, n_metrics


def render_metrics_panel(model_baseline, model_neurocx, customer_df, enc_meta, max_seq_len, device):
    st.markdown("### 📈 Evaluation Metrics (for Selected Customer Sequence)")
    
    b_metrics, n_metrics = compute_customer_metrics(
        model_baseline, model_neurocx, customer_df, enc_meta, max_seq_len, device
    )

    if b_metrics is None:
        st.info("Insufficient sequence steps to evaluate model performance.")
        return

    metric_keys = list(b_metrics.keys())

    # Table
    rows = []
    for mk in metric_keys:
        bv = b_metrics.get(mk, 0.0)
        nv = n_metrics.get(mk, 0.0)
        delta = nv - bv
        rows.append({
            "Metric": mk,
            "GRU Baseline": f"{bv:.4f}",
            "Neuro-CX": f"{nv:.4f}",
            "Δ (Neuro-CX − Baseline)": f"{'+' if delta >= 0 else ''}{delta:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Dynamic Plotly Chart
    chart_data = []
    for mk in metric_keys:
        chart_data.append({"Metric": mk, "Model": "GRU Baseline", "Score": b_metrics.get(mk, 0.0)})
        chart_data.append({"Metric": mk, "Model": "Neuro-CX", "Score": n_metrics.get(mk, 0.0)})
    df_chart = pd.DataFrame(chart_data)

    fig = px.bar(
        df_chart,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={"GRU Baseline": "#4c72b0", "Neuro-CX": "#dd8452"},
        text_auto=".3f"
    )
    fig.update_layout(
        title="Neuro-CX vs GRU Baseline — Evaluation Comparison on Selected Customer",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)"}
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main app ─────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_header()

    # Paths
    enc_path    = os.path.join(ROOT, "data", "encoders.json")
    inter_path  = os.path.join(ROOT, "data", "interactions_sample.parquet")

    enc_meta = load_encoders(enc_path)
    if enc_meta is None:
        st.error("❌ Encoder metadata not found. Run `py run_pipeline.py` first.")
        st.stop()

    ckpt_path, is_neuro_cx, top_k = render_sidebar(enc_meta)

    # Check checkpoints exist
    baseline_ckpt_path = os.path.join(ROOT, "checkpoints", "gru_baseline_best.pt")
    neurocx_ckpt_path = os.path.join(ROOT, "checkpoints", "neuro_cx_best.pt")

    if not os.path.exists(baseline_ckpt_path) or not os.path.exists(neurocx_ckpt_path):
        st.warning(
            "⚠️ Checkpoints not found in `checkpoints/`.\n\n"
            "Please run `py run_train.py` to train both models first."
        )
        st.stop()

    model_baseline, ckpt_baseline, device = load_model(baseline_ckpt_path)
    model_neurocx, ckpt_neurocx, _ = load_model(neurocx_ckpt_path)

    # Active model based on sidebar selection
    if is_neuro_cx:
        model = model_neurocx
        ckpt_data = ckpt_neurocx
    else:
        model = model_baseline
        ckpt_data = ckpt_baseline

    model_label = "🧠 Neuro-CX" if is_neuro_cx else "📊 GRU Baseline"

    # Reset if model changed
    prev_model = st.session_state.get("current_model")
    if prev_model != ckpt_path:
        reset_stepper()
        st.session_state.current_model = ckpt_path

    # ── Customer selector ─────────────────────────────────────────────────────
    interactions_df = load_sequences(inter_path)
    if interactions_df is None:
        st.error("❌ Interaction data not found. Run `py run_pipeline.py` first.")
        st.stop()

    all_customers = sorted(interactions_df["customer_id"].unique())
    n_items_total = enc_meta.get("n_items", 200)
    max_seq_len   = enc_meta.get("max_seq_len", 50)

    st.markdown("---")
    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        st.markdown("### 👤 Customer")
        chosen_cid = st.selectbox(
            "Select customer ID",
            all_customers,
            index=st.session_state.customer_idx,
            key="customer_selector",
        )
        new_idx = all_customers.index(chosen_cid)
        if new_idx != st.session_state.customer_idx:
            st.session_state.customer_idx = new_idx
            reset_stepper()

        customer_df = interactions_df[
            interactions_df["customer_id"] == chosen_cid
        ].sort_values("timestamp").reset_index(drop=True)

        n_events = len(customer_df)
        st.caption(f"{n_events} interaction events in history")
        st.caption(f"Model: **{model_label}**")

    with col_info:
        st.markdown("### 🗂️ Interaction History")
        display_df = customer_df[["action_type", "item_id", "dwell_time"]].copy()
        display_df["item"] = display_df["item_id"].apply(item_id_to_name)
        display_df["dwell_time"] = display_df["dwell_time"].apply(lambda x: f"{x:.0f}s")
        display_df = display_df.rename(columns={"action_type": "action", "dwell_time": "dwell"})
        display_df = display_df[["action", "item", "dwell"]]
        st.dataframe(display_df, use_container_width=True, height=180, hide_index=True)

    st.markdown("---")

    # ── Sequence Stepper ──────────────────────────────────────────────────────
    st.markdown("### 🎬 Sequence Stepper")

    step = st.session_state.step
    col_prev, col_step_info, col_next, col_reset = st.columns([1, 4, 1, 1])

    with col_prev:
        if st.button("◀ Prev", disabled=(step == 0), key="btn_prev"):
            st.session_state.step -= 1
            if st.session_state.hidden_history:
                st.session_state.hidden_history.pop()
            if st.session_state.action_history:
                st.session_state.action_history.pop()
            if st.session_state.item_history:
                st.session_state.item_history.pop()
            step = st.session_state.step

    with col_next:
        if st.button("Next ▶", disabled=(step >= n_events - 1), key="btn_next"):
            st.session_state.step += 1
            step = st.session_state.step

    with col_reset:
        if st.button("↺ Reset", key="btn_reset"):
            reset_stepper()
            step = 0

    step = st.session_state.step
    current_event = customer_df.iloc[step]
    action_str = str(current_event["action_type"]).strip()
    item_str   = item_id_to_name(int(current_event["item_id"]))

    with col_step_info:
        f"badge-{action_str.replace(' ', '_')}"
        icon = ACTION_ICONS.get(action_str, "•")
        color = ACTION_COLORS.get(action_str, "#aaa")
        st.markdown(
            f'<div class="step-info">'
            f'<b>Step {step + 1} / {n_events}</b> &nbsp;|&nbsp; '
            f'{icon} <span style="color:{color}"><b>{action_str.upper()}</b></span>'
            f' &nbsp;→&nbsp; 📦 {item_str}'
            f' &nbsp;|&nbsp; ⏱️ {current_event["dwell_time"]:.0f}s'
            f'</div>',
            unsafe_allow_html=True
        )

    # Progress bar
    st.progress((step + 1) / n_events)

    # ── Run inference at current step ─────────────────────────────────────────
    action_seq_t, item_seq_t, dwell_seq_t, weight_seq_t, _actual_len = \
        build_tensors_for_step(customer_df, step, enc_meta, max_seq_len, device)

    top_k_idx, top_k_scores, _last_hidden, _all_hiddens = get_recommendations(
        model, action_seq_t, item_seq_t, dwell_seq_t, weight_seq_t, top_k=top_k
    )

    # Update history for visualiser
    if action_str not in (st.session_state.action_history[-1:]
                           if st.session_state.action_history else []):
        pass  # always update on step change
    # Rebuild history cleanly each render (more reliable than incremental)
    hidden_hist = []
    action_hist = []
    for s in range(step + 1):
        a_t, i_t, d_t, w_t, _ = build_tensors_for_step(
            customer_df, s, enc_meta, max_seq_len, device
        )
        _, _, h_s, _ = get_recommendations(model, a_t, i_t, d_t, w_t, top_k=1)
        hidden_hist.append(h_s)
        action_hist.append(str(customer_df.iloc[s]["action_type"]).strip())

    # ── Main content columns ──────────────────────────────────────────────────
    col_recs, col_viz = st.columns([1, 2])

    with col_recs:
        render_recommendations(top_k_idx, top_k_scores, n_items_total)

        # Show signal weight
        aw = enc_meta.get("action_weights", {})
        w_val = aw.get(action_str, 1.0)
        st.markdown(f"**Signal weight:** `{w_val}` "
                    f"({'Strong reinforcement' if w_val >= 2 else 'Moderate' if w_val >= 1 else 'Weak signal'})")

        if is_neuro_cx:
            import math
            decay_hl = ckpt_data.get("model_config", {}).get("decay_half_life", 10.0)
            lam = math.log(2) / decay_hl
            decay_val = math.exp(-lam / max(w_val, 0.1))
            st.markdown(f"**Decay factor:** `{decay_val:.3f}` "
                        f"({'Low forgetting' if decay_val > 0.8 else 'Moderate forgetting' if decay_val > 0.5 else 'High forgetting'})")

    with col_viz:
        render_hidden_state_viz(hidden_hist, action_hist)

    # ── Metrics panel ─────────────────────────────────────────────────────────
    st.markdown("---")
    render_metrics_panel(model_baseline, model_neurocx, customer_df, enc_meta, max_seq_len, device)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>Neuro-CX Research Prototype · MSc Data Science · "
        "All sequences are synthetic/derived · No real user data.</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
