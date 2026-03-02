import streamlit as st
import pandas as pd
import numpy as np
import json, pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — Dark Cyberpunk Theme ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── Base ── */
html, body {
    font-family: 'Exo 2', sans-serif;
    background-color: #060910;
    color: #e2e8f0;
}
.stApp { background: #060910; }

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,255,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1a 0%, #060910 100%);
    border-right: 1px solid rgba(0,255,180,0.15);
}
/* ── Sidebar radio ── */
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #00ffc8 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1422 0%, #111827 100%);
    border: 1px solid rgba(0,255,180,0.12);
    border-radius: 12px;
    padding: 18px 20px !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,255,180,0.35);
    transform: translateY(-2px);
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00ffc8, #0080ff);
    border-radius: 2px 0 0 2px;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: #00ffc8 !important;
    font-size: 1.7rem !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Section headers ── */
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid rgba(0,255,180,0.15);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Page title ── */
.page-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00ffc8 0%, #0080ff 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.page-subtitle {
    color: #475569;
    font-size: 0.9rem;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 24px;
}

/* ── Verdict cards ── */
.verdict-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f87171;
    letter-spacing: 0.1em;
}
.verdict-legit {
    background: linear-gradient(135deg, rgba(0,255,180,0.12), rgba(0,255,180,0.04));
    border: 1px solid rgba(0,255,180,0.35);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00ffc8;
    letter-spacing: 0.1em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #00ffc8, #0080ff) !important;
    color: #060910 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs / Selects ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background: #0d1422 !important;
    border-color: rgba(0,255,180,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Divider ── */
hr { border-color: rgba(0,255,180,0.08) !important; }

/* ── Plotly charts bg ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Sidebar logo/brand ── */
.sidebar-brand {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00ffc8, #0080ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.05em;
    padding: 8px 0 4px 0;
}
.sidebar-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #334155;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
}

/* ── Info box ── */
.info-box {
    background: rgba(0,128,255,0.08);
    border: 1px solid rgba(0,128,255,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #93c5fd;
    font-family: 'Share Tech Mono', monospace;
    margin: 12px 0;
}

/* ── Architecture table ── */
.arch-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
}
.arch-table th {
    background: rgba(0,255,180,0.08);
    color: #00ffc8;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid rgba(0,255,180,0.2);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem;
}
.arch-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #94a3b8;
}
.arch-table tr:hover td { background: rgba(0,255,180,0.04); color: #e2e8f0; }

/* Hide Streamlit top header */
header[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
}
/* Hide scroll-to-top floating control safely */
[data-testid="stScrollToTopButton"] {
    display: none !important;
}
button[aria-label*="Scroll"] {
    display: none !important;
}                        
</style>
""", unsafe_allow_html=True)

# ── Plotly dark layout helper ─────────────────────────────────
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor='rgba(13,20,34,0)',
        plot_bgcolor='rgba(13,20,34,0)',
        font=dict(family='Exo 2, sans-serif', color='#94a3b8', size=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.08)'),
        margin=dict(t=30, b=30, l=10, r=10)
    )
    base.update(kwargs)
    return base

COLORS = {
    'teal':   '#00ffc8',
    'blue':   '#0080ff',
    'purple': '#a855f7',
    'red':    '#f87171',
    'orange': '#fb923c',
    'green':  '#4ade80',
    'grid':   'rgba(255,255,255,0.04)'
}

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fraud_ann_model.keras")

@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)

@st.cache_resource
def load_preprocessing():
    with open("scaler.pkl",        "rb") as f: scaler   = pickle.load(f)
    with open("encoders.pkl",      "rb") as f: encoders = pickle.load(f)
    with open("feature_names.pkl", "rb") as f: features = pickle.load(f)
    return scaler, encoders, features

try:
    model   = load_model()
    metrics = load_metrics()
    scaler, encoders, feature_names = load_preprocessing()
except Exception as e:
    st.error(f"⚠️ Run `python model.py` first!\n\n{e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛡️ FRAUD SHIELD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">UPI TRANSACTION MONITOR</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊  Dashboard",
            "🔍  Predict Transaction",
            "📈  Model Performance",
            "⚙️  Fine-Tune Model"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-family:Share Tech Mono;font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:0.12em;">System Status</p>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    col_a.metric("AUC", f"{metrics['roc_auc']}")
    col_b.metric("F1", f"{metrics['f1']}")
    col_a.metric("Records", f"{metrics['total_transactions']:,}")
    col_b.metric("Fraud%", f"{metrics['fraud_rate']}%")

    st.markdown("---")
    st.markdown(
        '<p style="font-family:Share Tech Mono;font-size:0.65rem;color:#1e293b;text-align:center;letter-spacing:0.1em;">POWERED BY ANN + SMOTE</p>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown('<div class="page-title">Transaction Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">> REAL-TIME UPI FRAUD DETECTION SYSTEM — ANN MODEL v1.0</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Transactions",  f"{metrics['total_transactions']:,}")
    k2.metric("Fraud Detected",      f"{metrics['fraud_count']:,}",
              delta=f"↑ {metrics['fraud_rate']}%", delta_color="inverse")
    k3.metric("Legitimate",          f"{metrics['legit_count']:,}")
    k4.metric("ROC-AUC Score",       f"{metrics['roc_auc']}")
    k5.metric("F1 Score",            f"{metrics['f1']}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.markdown('<div class="section-title">Transaction Split</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[metrics['legit_count'], metrics['fraud_count']],
            hole=0.65,
            marker=dict(
                colors=[COLORS['teal'], COLORS['red']],
                line=dict(color='#060910', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(family='Exo 2', size=13, color='#e2e8f0'),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
        ))
        fig.add_annotation(
            text=f"<b>{metrics['fraud_rate']}%</b><br><span style='font-size:10px'>FRAUD RATE</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['red'], family='Rajdhani')
        )
        fig.update_layout(**dark_layout(
        height=320,
        showlegend=True,
        legend=dict(
            orientation='h',
            y=-0.05,
            x=0.5,
            xanchor='center',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.08)'
        )
))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics['confusion_matrix']
        labels = [["True Negative", "False Positive"],
                  ["False Negative", "True Positive"]]
        vals   = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
        colors_cm = [[COLORS['teal'], COLORS['orange']], [COLORS['orange'], COLORS['teal']]]

        fig = go.Figure(go.Heatmap(
            z=vals,
            x=["Predicted: Legit", "Predicted: Fraud"],
            y=["Actual: Legit",    "Actual: Fraud"],
            text=[[f"<b>{vals[i][j]:,}</b><br><span style='font-size:9px'>{labels[i][j]}</span>"
                   for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            colorscale=[[0, 'rgba(0,255,200,0.15)'], [1, 'rgba(248,113,113,0.25)']],
            showscale=False,
            hovertemplate='%{x}<br>%{y}<br>Count: %{z:,}<extra></extra>'
        ))
        fig.update_layout(**dark_layout(
        height=320,
        xaxis=dict(
            gridcolor=COLORS['grid'],
            title='False Positive Rate'   # keep whatever title you had
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            title='True Positive Rate'    # keep your original title
        )
))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='rgba(100,116,139,0.4)', dash='dash', width=1),
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=metrics['fpr'], y=metrics['tpr'],
            mode='lines',
            name=f'ANN Model  (AUC = {metrics["roc_auc"]})',
            line=dict(color=COLORS['teal'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0,255,200,0.06)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        fig.update_layout(**dark_layout(
        height=320,
        xaxis=dict(
            title='False Positive Rate',
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='True Positive Rate',
            gridcolor=COLORS['grid']
        ),
        legend=dict(
            x=0.4,
            y=0.05,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.08)'
        )
))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Training Loss Curve</div>', unsafe_allow_html=True)
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=metrics['train_loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color=COLORS['blue'], width=2),
            hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=metrics['val_loss'],
            mode='lines',
            name='Val Loss',
            line=dict(color=COLORS['purple'], width=2, dash='dot'),
            hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))
        fig.update_layout(**dark_layout(
        height=320,
        xaxis=dict(
            title='Epoch',
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='Binary CE Loss',
            gridcolor=COLORS['grid']
        ),
        legend=dict(
            x=0.6,
            y=0.9,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.08)'
        )
))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Predict Transaction":
    st.markdown('<div class="page-title">Transaction Risk Analyser</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">> ENTER TRANSACTION PARAMETERS TO RUN FRAUD INFERENCE</div>', unsafe_allow_html=True)

    banks      = ['SBI','HDFC','ICICI','Axis','PNB','Kotak','Yes Bank','Bank of Baroda']
    categories = ['Grocery','Electronics','Fuel','Food','Entertainment','Healthcare','Recharge','Shopping']
    tx_types   = ['P2P','P2M','P2B']
    devices    = ['Android','iOS','Web']
    states     = ['Delhi','Maharashtra','Karnataka','Tamil Nadu','Uttar Pradesh','Gujarat','Rajasthan','West Bengal']

    st.markdown('<div class="section-title">Transaction Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        amount     = st.number_input("Amount (INR)",           min_value=1.0,   max_value=200000.0, value=5000.0, step=100.0)
        avg_amount = st.number_input("User Avg Amount (INR)",  min_value=1.0,   max_value=100000.0, value=3000.0, step=100.0)
        hour       = st.slider("Hour of Transaction (0–23)",   0, 23, 14)
        failed_att = st.slider("Failed Attempts",              0, 5, 0)

    with col2:
        freq       = st.slider("Transaction Frequency / day",  1, 30, 3)
        is_new_dev = st.selectbox("New Device?",               ["No", "Yes"])
        is_unusual = st.selectbox("Unusual Location?",         ["No", "Yes"])
        is_night   = st.selectbox("Night Transaction?",        ["No", "Yes"])

    with col3:
        sender_bank  = st.selectbox("Sender Bank",             banks)
        merchant_cat = st.selectbox("Merchant Category",       categories)
        tx_type      = st.selectbox("Transaction Type",        tx_types)
        device_type  = st.selectbox("Device Type",             devices)
        sender_state = st.selectbox("Sender State",            states)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  ANALYSE TRANSACTION", use_container_width=True):
        input_dict = {
            'Amount':               amount,
            'Hour':                 hour,
            'AvgUserAmount':        avg_amount,
            'FailedAttempts':       failed_att,
            'IsNewDevice':          1 if is_new_dev == "Yes" else 0,
            'IsUnusualLocation':    1 if is_unusual == "Yes" else 0,
            'TransactionFrequency': freq,
            'SenderBank':           sender_bank,
            'MerchantCategory':     merchant_cat,
            'TransactionType':      tx_type,
            'DeviceType':           device_type,
            'SenderState':          sender_state,
            'AmountToAvgRatio':     round(amount / (avg_amount + 1), 4),
            'IsNightHour':          1 if is_night == "Yes" else 0
        }

        input_df = pd.DataFrame([input_dict])
        for col in ["SenderBank","MerchantCategory","TransactionType","DeviceType","SenderState"]:
            if col in encoders:
                try:    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                except: input_df[col] = 0

        input_df     = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        prob         = float(model.predict(input_scaled, verbose=0).flatten()[0])
        threshold    = metrics.get('best_threshold', 0.5)
        is_fraud     = prob >= threshold

        st.markdown("---")
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fraud Probability", f"{prob*100:.2f}%")
        r2.metric("Risk Threshold",    f"{int(threshold*100)}%")
        r3.metric("Decision",          "FRAUD 🚨" if is_fraud else "SAFE ✅")
        r4.metric("Confidence",        f"{abs(prob - threshold)*100:.1f}% margin")

        verdict_html = (
            f'<div class="verdict-fraud">🚨 &nbsp; FRAUDULENT TRANSACTION DETECTED &nbsp; — &nbsp; {prob*100:.1f}% RISK</div>'
            if is_fraud else
            f'<div class="verdict-legit">✅ &nbsp; TRANSACTION CLEARED &nbsp; — &nbsp; {prob*100:.1f}% RISK</div>'
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        gc1, gc2 = st.columns([1, 1])

        with gc1:
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                delta={'reference': threshold * 100, 'relative': False,
                       'valueformat': '.1f',
                       'increasing': {'color': COLORS['red']},
                       'decreasing': {'color': COLORS['teal']}},
                title={'text': "FRAUD RISK SCORE", 'font': {'family': 'Rajdhani', 'size': 14, 'color': '#64748b'}},
                number={'suffix': '%', 'font': {'family': 'Rajdhani', 'size': 40,
                        'color': COLORS['red'] if is_fraud else COLORS['teal']}},
                gauge={
                    'axis': {'range': [0,100], 'tickfont': {'color': '#475569', 'size': 10},
                             'tickcolor': '#1e293b'},
                    'bar':  {'color': COLORS['red'] if is_fraud else COLORS['teal'], 'thickness': 0.25},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'bordercolor': 'rgba(255,255,255,0.05)',
                    'steps': [
                        {'range': [0,  40],  'color': 'rgba(74,222,128,0.08)'},
                        {'range': [40, 70],  'color': 'rgba(251,146,60,0.08)'},
                        {'range': [70, 100], 'color': 'rgba(248,113,113,0.12)'}
                    ],
                    'threshold': {
                        'line': {'color': '#f59e0b', 'width': 2},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Exo 2'),
                height=300, margin=dict(t=20, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with gc2:
            # Feature risk bar
            risk_features = {
                'High Amount':          min(amount / 50000, 1.0),
                'Amount vs Avg':        min((amount / (avg_amount + 1)) / 10, 1.0),
                'Failed Attempts':      failed_att / 5,
                'New Device':           1.0 if is_new_dev == "Yes" else 0.1,
                'Unusual Location':     1.0 if is_unusual == "Yes" else 0.1,
                'Night Transaction':    1.0 if is_night == "Yes" else 0.1,
                'High Frequency':       min(freq / 30, 1.0),
            }
            rf_names  = list(risk_features.keys())
            rf_values = [round(v * 100, 1) for v in risk_features.values()]
            rf_colors = [COLORS['red'] if v > 60 else COLORS['orange'] if v > 30 else COLORS['teal']
                         for v in rf_values]

            fig = go.Figure(go.Bar(
                x=rf_values, y=rf_names,
                orientation='h',
                marker=dict(color=rf_colors, line=dict(width=0)),
                text=[f"{v}%" for v in rf_values],
                textposition='outside',
                textfont=dict(family='Share Tech Mono', size=10, color='#94a3b8'),
                hovertemplate='%{y}<br>Risk: %{x}%<extra></extra>'
            ))
            fig.update_layout(
                **dark_layout(),
                title=dict(text='Risk Factor Breakdown', font=dict(size=13, color='#64748b', family='Rajdhani'), x=0),
                height=300,
                xaxis=dict(range=[0,120], showgrid=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=10, family='Share Tech Mono')),
                margin=dict(t=35, b=10, l=10, r=60)
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown('<div class="page-title">Model Performance Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">> ANN EVALUATION METRICS — TEST SET RESULTS</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",   f"{metrics['accuracy']*100:.2f}%")
    m2.metric("Precision",  f"{metrics['precision']*100:.2f}%")
    m3.metric("Recall",     f"{metrics['recall']*100:.2f}%")
    m4.metric("F1 Score",   f"{metrics['f1']}")
    m5.metric("ROC-AUC",    f"{metrics['roc_auc']}")

    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        st.markdown('<div class="section-title">Metrics Overview</div>', unsafe_allow_html=True)
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        metric_vals  = [metrics['accuracy'], metrics['precision'],
                        metrics['recall'],   metrics['f1'], metrics['roc_auc']]
        bar_colors   = [COLORS['teal'], COLORS['blue'], COLORS['purple'],
                        COLORS['orange'], COLORS['green']]

        fig = go.Figure(go.Bar(
            x=metric_names, y=metric_vals,
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                opacity=0.9
            ),
            text=[f"{v:.3f}" for v in metric_vals],
            textposition='outside',
            textfont=dict(family='Rajdhani', size=13, color='#e2e8f0'),
            hovertemplate='%{x}<br>Value: %{y:.4f}<extra></extra>',
            width=0.55
        ))
        fig.update_layout(**dark_layout(
        height=340,
        yaxis=dict(
            range=[0, 1.12],
            gridcolor=COLORS['grid'],
            tickformat='.0%'
        ),
        xaxis=dict(
            showgrid=False
        ),
        showlegend=False
))
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        st.markdown('<div class="section-title">Training vs Validation Accuracy</div>', unsafe_allow_html=True)
        epochs = list(range(1, len(metrics['train_acc']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=metrics['train_acc'],
            mode='lines',
            name='Train Accuracy',
            line=dict(color=COLORS['teal'], width=2),
            fill='tozeroy', fillcolor='rgba(0,255,200,0.04)',
            hovertemplate='Epoch %{x}<br>Acc: %{y:.4f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=metrics['val_acc'],
            mode='lines',
            name='Val Accuracy',
            line=dict(color=COLORS['purple'], width=2, dash='dot'),
            hovertemplate='Epoch %{x}<br>Acc: %{y:.4f}<extra></extra>'
        ))
        fig.update_layout(**dark_layout(
        height=340,
        xaxis=dict(
            title='Epoch',
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='Accuracy',
            gridcolor=COLORS['grid']
        ),
        legend=dict(
            x=0.55,
            y=0.05,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.08)'
        )
))
        st.plotly_chart(fig, use_container_width=True)

    # Architecture
    st.markdown("---")
    st.markdown('<div class="section-title">ANN Architecture & Config</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
<table class="arch-table">
  <tr><th>Layer</th><th>Units</th><th>Activation</th><th>Regularisation</th></tr>
  <tr><td>Input</td><td>14</td><td>—</td><td>—</td></tr>
  <tr><td>Dense + BatchNorm</td><td>128</td><td>ReLU</td><td>Dropout 0.3</td></tr>
  <tr><td>Dense + BatchNorm</td><td>64</td><td>ReLU</td><td>Dropout 0.3</td></tr>
  <tr><td>Dense</td><td>32</td><td>ReLU</td><td>Dropout 0.2</td></tr>
  <tr><td>Output</td><td>1</td><td>Sigmoid</td><td>—</td></tr>
</table>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
<div class="info-box">
OPTIMIZER &nbsp;&nbsp;&nbsp; Adam (lr=0.001)<br>
LOSS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Binary Crossentropy<br>
BATCH SIZE &nbsp;&nbsp; 64<br>
MAX EPOCHS &nbsp;&nbsp; 50 (EarlyStopping)<br>
IMBALANCE &nbsp;&nbsp;&nbsp; SMOTE 1:1 Resampling<br>
THRESHOLD &nbsp;&nbsp;&nbsp; {metrics['best_threshold']} (tuned on F1)<br>
TOTAL PARAMS &nbsp; 12,801
</div>
        """, unsafe_allow_html=True)

    # Precision/Recall tradeoff
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification Summary — Test Set</div>', unsafe_allow_html=True)

    cm = metrics['confusion_matrix']
    summary_data = {
        "Metric":    ["True Negatives", "False Positives", "False Negatives", "True Positives"],
        "Count":     [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
        "Meaning":   ["Legit correctly cleared", "Legit incorrectly flagged",
                      "Fraud missed by model",  "Fraud correctly caught"],
        "Status":    ["✅ Good", "⚠️ FP Cost", "❌ Risk", "✅ Caught"]
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )
    st.markdown("""
<hr style="margin-top:40px;border-color:rgba(0,255,180,0.08);">

<div style="
    text-align:center;
    font-family:'Share Tech Mono', monospace;
    font-size:0.75rem;
    color:#334155;
    letter-spacing:0.12em;
    padding:12px 0;
">
BUILT BY <span style="color:#00ffc8;">AMAN</span> — COLLEGE ANN PROJECT
</div>
""", unsafe_allow_html=True)
    
# ══════════════════════════════════════════════════════════════
# PAGE 4 — FINE TUNE MODEL
# ══════════════════════════════════════════════════════════════
elif page == "⚙️  Fine-Tune Model":

    st.markdown('<div class="page-title">Model Fine-Tuning Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">> DECISION BOUNDARY & COST OPTIMIZATION</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Decision Threshold Control</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Adjust Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(metrics.get("best_threshold", 0.5)),
        step=0.01
    )

    y_test = np.array(metrics["y_test"])
    y_probs = np.array(metrics["y_probs"])

    y_pred_new = (y_probs >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    cm_new = confusion_matrix(y_test, y_pred_new)
    precision_new = precision_score(y_test, y_pred_new, zero_division=0)
    recall_new = recall_score(y_test, y_pred_new, zero_division=0)
    f1_new = f1_score(y_test, y_pred_new, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision_new:.3f}")
    m2.metric("Recall", f"{recall_new:.3f}")
    m3.metric("F1 Score", f"{f1_new:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Updated Confusion Matrix</div>', unsafe_allow_html=True)

    labels = [["True Negative", "False Positive"],
              ["False Negative", "True Positive"]]

    fig = go.Figure(go.Heatmap(
        z=cm_new,
        x=["Predicted: Legit", "Predicted: Fraud"],
        y=["Actual: Legit", "Actual: Fraud"],
        text=[[f"<b>{cm_new[i][j]:,}</b><br><span style='font-size:9px'>{labels[i][j]}</span>"
               for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        colorscale=[[0, 'rgba(0,255,200,0.15)'], [1, 'rgba(248,113,113,0.25)']],
        showscale=False,
        hovertemplate='%{x}<br>%{y}<br>Count: %{z:,}<extra></extra>'
    ))

    fig.update_layout(**dark_layout(height=320))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="info-box">
Lower threshold → Higher Recall (catches more fraud) but more False Positives.  
Higher threshold → Higher Precision but risks missing fraud.
</div>
""", unsafe_allow_html=True)
    
# ──────────────────────────────────────────────────────────────
# GLOBAL FOOTER (ALL PAGES)
# ──────────────────────────────────────────────────────────────

st.markdown("""
<hr style="margin-top:40px;border-color:rgba(0,255,180,0.08);">

<div style="
    text-align:center;
    font-family:'Share Tech Mono', monospace;
    font-size:0.75rem;
    color:#334155;
    letter-spacing:0.12em;
    padding:12px 0;
">
BUILT BY <span style="color:#00ffc8;">AMAN</span> — ANN PROJECT
</div>
""", unsafe_allow_html=True)
