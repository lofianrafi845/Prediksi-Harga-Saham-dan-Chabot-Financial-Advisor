import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import getpass

# Check if groq is installed
try:
    from groq import Groq
    groq_available = True
except ImportError:
    groq_available = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="StockMaster AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ─────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border: 1px solid #1e40af33;
        box-shadow: 0 4px 32px rgba(59,130,246,0.15);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0;
        font-size: 0.9rem;
    }

    /* ── Metric cards ────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 0.5rem;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-label {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.3rem;
        font-weight: 700;
        word-break: break-all;
    }
    .metric-delta-up   { color: #10b981; font-size: 0.78rem; }
    .metric-delta-down { color: #ef4444; font-size: 0.78rem; }

    /* ── Section headers ─────────────────────────── */
    .section-header {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin: 1.2rem 0 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1e40af44;
    }

    /* ── Chat bubbles ────────────────────────────── */
    .chat-bubble-user {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        color: #f8fafc;
        border-radius: 16px 16px 4px 16px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        max-width: 90%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(30,64,175,0.3);
        font-size: 0.88rem;
        word-wrap: break-word;
    }
    .chat-bubble-bot {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 16px 16px 16px 4px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        max-width: 95%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-size: 0.88rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .chat-role-user { text-align: right; color: #60a5fa; font-size: 0.68rem; margin-bottom: 0.2rem; }
    .chat-role-bot  { color: #34d399; font-size: 0.68rem; margin-bottom: 0.2rem; }

    /* ── Colors ──────────────────────────────────── */
    .prediction-up   { color: #10b981; font-weight: 600; }
    .prediction-down { color: #ef4444; font-weight: 600; }

    /* ── Buttons ─────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30,64,175,0.4);
    }

    /* ── Badges ──────────────────────────────────── */
    .info-badge {
        display: inline-block;
        background: #1e3a5f44;
        border: 1px solid #1e40af55;
        color: #60a5fa;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    .sidebar-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] label { color: #cbd5e1 !important; }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.5rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* dark theme for plotly */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Responsive: tablet (max 768px) ─────────── */
    @media (max-width: 768px) {
        .main-header { padding: 1rem 1.2rem; border-radius: 10px; }
        .main-header h1 { font-size: 1.3rem; }
        .main-header p { font-size: 0.8rem; }

        .metric-value { font-size: 1rem; }
        .metric-label { font-size: 0.65rem; }

        .section-header { font-size: 0.9rem; }

        .chat-bubble-user, .chat-bubble-bot {
            max-width: 100%;
            font-size: 0.84rem;
        }

        /* Stack columns vertically on mobile */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* Reduce plotly chart height on mobile */
        .js-plotly-plot { min-height: 280px; }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.78rem;
            padding: 0.4rem 0.5rem;
        }
    }

    /* ── Responsive: small phone (max 480px) ────── */
    @media (max-width: 480px) {
        .main-header h1 { font-size: 1.1rem; }
        .metric-value { font-size: 0.9rem; }
        .info-badge { font-size: 0.65rem; padding: 0.1rem 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
@st.cache_data
def generate_demo_data():
    """Generate realistic synthetic IDX stock data for demo."""
    np.random.seed(42)
    tickers = ['BBCA', 'BBRI', 'TLKM', 'BMRI', 'ASII', 'UNVR', 'GOTO', 'ADRO', 'ANTM', 'PTBA']
    
    end   = datetime.today()
    start = end - timedelta(days=365 * 3)
    dates = pd.date_range(start, end, freq='B')
    
    # Base prices (realistic IDX prices)
    base = {
        'BBCA': 9500, 'BBRI': 5400, 'TLKM': 3800, 'BMRI': 7100,
        'ASII': 5200, 'UNVR': 2500, 'GOTO': 62, 'ADRO': 2800,
        'ANTM': 1650, 'PTBA': 2900,
    }
    
    rows = []
    for ticker in tickers:
        price = base[ticker]
        eps_base  = np.random.uniform(200, 800)
        der_base  = np.random.uniform(0.3, 2.5)
        roa_base  = np.random.uniform(0.02, 0.15)
        roe_base  = np.random.uniform(0.10, 0.30)
        prices_arr = [price]
        
        for i in range(1, len(dates)):
            drift = np.random.normal(0.0003, 0.015)
            price = max(price * (1 + drift), 50)
            prices_arr.append(price)
        
        for i, (d, p) in enumerate(zip(dates, prices_arr)):
            ma5  = np.mean(prices_arr[max(0, i-4):i+1])
            ma20 = np.mean(prices_arr[max(0, i-19):i+1])
            vol20 = np.std(prices_arr[max(0, i-19):i+1]) / (p + 1e-9)
            rows.append({
                'Tanggal':      d,
                'Ticker':       ticker,
                'Close_Price':  round(p, 0),
                'Basic EPS':    round(eps_base + np.random.normal(0, 20), 2),
                'DER':          round(der_base + np.random.normal(0, 0.05), 4),
                'ROA':          round(roa_base + np.random.normal(0, 0.002), 4),
                'ROE':          round(roe_base + np.random.normal(0, 0.005), 4),
                'MA5':          round(ma5, 2),
                'MA20':         round(ma20, 2),
                'Volatility20': round(vol20, 4),
                'Target':       round(prices_arr[min(i+1, len(prices_arr)-1)], 0),
            })
    return pd.DataFrame(rows)

def simulate_xgb_prediction(df_ticker, future_days=30):
    """Simulate XGBoost-style predictions (deterministic from data)."""
    prices = df_ticker['Close_Price'].values
    last   = prices[-1]
    trend  = (prices[-1] - prices[-20]) / 20 if len(prices) >= 20 else 0
    vol    = np.std(np.diff(prices[-20:])) if len(prices) >= 21 else last * 0.01
    preds  = []
    p = last
    for i in range(future_days):
        noise = np.random.normal(trend, vol * 0.4)
        p = p + noise
        preds.append(max(p, 1))
    return np.array(preds)

def simulate_lstm_prediction(df_ticker, future_days=30):
    """Simulate LSTM-style predictions (slightly smoother)."""
    prices = df_ticker['Close_Price'].values
    last   = prices[-1]
    ma20   = df_ticker['MA20'].values[-1]
    trend  = (last - ma20) / 20
    vol    = df_ticker['Volatility20'].values[-1] * last
    preds  = []
    p = last
    for i in range(future_days):
        momentum = trend * 0.8
        noise    = np.random.normal(momentum, vol * 0.3)
        p = p + noise
        preds.append(max(p, 1))
    return np.array(preds)

def call_groq_api(messages: list, system_prompt: str) -> str:
    """Call Groq API via groq."""
    if not groq_available:
        return "groq belum terinstal. Jalankan: pip install groq"
    try:
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            return "Masukkan Groq API Key di sidebar untuk mengaktifkan chatbot."
        
        client = Groq(api_key=api_key)
        
        # Build messages list for Groq API
        groq_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            # Map role
            role = 'assistant' if m['role'] in ['assistant', 'model'] else 'user'
            groq_messages.append({'role': role, 'content': m['content']})
            
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error API: {str(e)}"

def build_rag_context(df: pd.DataFrame, ticker: str, question: str) -> str:
    """Build relevant context from stock data for RAG."""
    df_t = df[df['Ticker'] == ticker].sort_values('Tanggal')
    if df_t.empty:
        return "Tidak ada data untuk ticker ini."
    
    latest = df_t.iloc[-1]
    prev5  = df_t.tail(5)
    
    ctx_parts = [
        f"SAHAM: {ticker}",
        f"Harga penutupan terbaru: Rp {latest['Close_Price']:,.0f} (per {latest['Tanggal'].strftime('%d %B %Y')})",
        f"MA5: {latest['MA5']:,.2f} | MA20: {latest['MA20']:,.2f}",
        f"Trend: {'Bullish ↑' if latest['MA5'] > latest['MA20'] else 'Bearish ↓'} (MA5 {'>' if latest['MA5'] > latest['MA20'] else '<'} MA20)",
        f"Volatilitas 20-hari: {latest['Volatility20']:.4f}",
        f"EPS: {latest['Basic EPS']:.2f} | DER: {latest['DER']:.4f} | ROA: {latest['ROA']*100:.2f}% | ROE: {latest['ROE']*100:.2f}%",
        "",
        "5 hari terakhir:",
    ]
    for _, row in prev5.iterrows():
        ctx_parts.append(f"  - {row['Tanggal'].strftime('%d %b %Y')}: Rp {row['Close_Price']:,.0f}")
    
    # Add comparison with other tickers if multiple in df
    all_tickers = df.groupby('Ticker').last()[['ROE', 'ROA', 'Close_Price']].reset_index()
    ctx_parts.append("\nPerbandingan ROE semua saham:")
    for _, row in all_tickers.sort_values('ROE', ascending=False).head(5).iterrows():
        mark = " ← (dipilih)" if row['Ticker'] == ticker else ""
        ctx_parts.append(f"  {row['Ticker']}: ROE {row['ROE']*100:.2f}%{mark}")
    
    return "\n".join(ctx_parts)

# ── Session state init ────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_stock_data():
    csv_file = "stockmaster_preprocessed.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            return df
        except Exception as e:
            st.error(f"Error loading CSV data: {e}")
            return generate_demo_data()
    else:
        st.warning(f"File {csv_file} tidak ditemukan. Menggunakan data demo.")
        return generate_demo_data()

df = load_stock_data()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">StockMaster AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key (Groq)**")
    if not groq_available:
        st.error("Package groq belum terinstal.")
        st.info("Jalankan `pip install groq` untuk mengaktifkan fitur ini.")
    else:
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key tersimpan")
        else:
            st.info("Masukkan API key untuk aktifkan chatbot")
    
    st.markdown("---")
    st.markdown("**Pilih Saham**")
    tickers_list = sorted(df['Ticker'].unique().tolist())
    selected_ticker = st.selectbox("Ticker", tickers_list, index=0, label_visibility="collapsed")
    
    st.markdown("**Model Prediksi**")
    model_choice = st.radio(
        "Model",
        ["XGBoost", "LSTM", "Ensemble (XGB + LSTM)"],
        label_visibility="collapsed",
    )
    
    st.markdown("**Rentang Prediksi**")
    future_days = st.slider("Hari ke depan", 7, 90, 30, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown('<span class="info-badge">IDX Saham</span> <span class="info-badge">RAG Chatbot</span>', unsafe_allow_html=True)
    st.markdown('<span class="info-badge">XGBoost</span> <span class="info-badge">LSTM</span>', unsafe_allow_html=True)

# ── Main header ───────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>StockMaster AI</h1>
    <p>Prediksi Harga Saham Indonesia &amp; Financial Advisor — IDX Bursa Efek Indonesia</p>
</div>
""", unsafe_allow_html=True)

# ── Ticker quick metrics ───────────────────────────────────────
df_sel = df[df['Ticker'] == selected_ticker].sort_values('Tanggal')
latest = df_sel.iloc[-1]
prev   = df_sel.iloc[-2] if len(df_sel) > 1 else latest

delta_price = latest['Close_Price'] - prev['Close_Price']
delta_pct   = (delta_price / prev['Close_Price'] * 100) if prev['Close_Price'] else 0
arrow = "▲" if delta_price >= 0 else "▼"
delta_cls = "metric-delta-up" if delta_price >= 0 else "metric-delta-down"

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Harga Terakhir", f"Rp {latest['Close_Price']:,.0f}", f"{arrow} {abs(delta_pct):.2f}%", delta_cls),
    ("MA5",  f"{latest['MA5']:,.0f}", "Moving Avg 5-hari", "metric-label"),
    ("MA20", f"{latest['MA20']:,.0f}", "Moving Avg 20-hari", "metric-label"),
    ("ROE",  f"{latest['ROE']*100:.2f}%", "Return on Equity", "metric-label"),
    ("DER",  f"{latest['DER']:.3f}", "Debt-to-Equity Ratio", "metric-label"),
]
for col, (lbl, val, sub, sub_cls) in zip([c1,c2,c3,c4,c5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
            <div class="{sub_cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediksi Harga", "Financial Advisor", "Data & Fundamental"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Prediksi
# ══════════════════════════════════════════════════════════════
with tab1:
    import plotly.graph_objects as go
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(f'<div class="section-header">Riwayat & Prediksi Harga — {selected_ticker}</div>', unsafe_allow_html=True)
        
        # Historical data (last 180 days)
        df_hist = df_sel.tail(180)
        hist_dates  = df_hist['Tanggal'].tolist()
        hist_prices = df_hist['Close_Price'].tolist()
        hist_ma5    = df_hist['MA5'].tolist()
        hist_ma20   = df_hist['MA20'].tolist()
        
        # Future dates
        last_date  = df_hist['Tanggal'].max()
        fut_dates = []
        current_date = last_date
        while len(fut_dates) < future_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                fut_dates.append(current_date)
        
        # Generate predictions
        np.random.seed(int(time.time()) % 100)
        xgb_preds  = simulate_xgb_prediction(df_hist, future_days)
        lstm_preds = simulate_lstm_prediction(df_hist, future_days)
        ens_preds  = (xgb_preds * 0.5 + lstm_preds * 0.5)
        
        fig = go.Figure()

        # ── Historical line (solid blue) ────────────────────────
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            mode='lines', name='Historical Data',
            line=dict(color='#1a56ff', width=2),
            hovertemplate='%{x|%d %b %Y}<br>Price: %{y:,.0f}<extra></extra>',
        ))

        # ── Prediction line (dashed yellow) ─────────────────────
        all_pred_x = [last_date] + fut_dates

        if model_choice == "XGBoost":
            all_pred_y = [hist_prices[-1]] + xgb_preds.tolist()
            pred_label = 'Prediction'
        elif model_choice == "LSTM":
            all_pred_y = [hist_prices[-1]] + lstm_preds.tolist()
            pred_label = 'Prediction'
        else:
            all_pred_y = [hist_prices[-1]] + ens_preds.tolist()
            pred_label = 'Prediction'

        fig.add_trace(go.Scatter(
            x=all_pred_x, y=all_pred_y,
            mode='lines', name=pred_label,
            line=dict(color='#f5c518', width=2.5, dash='dash'),
            hovertemplate='%{x|%d %b %Y}<br>Predicted: %{y:,.0f}<extra></extra>',
        ))

        # ── Vertical dashed red separator line ──────────────────
        y_min = min(hist_prices + all_pred_y)
        y_max = max(hist_prices + all_pred_y)
        y_pad = (y_max - y_min) * 0.05

        fig.add_shape(
            type='line',
            x0=last_date, x1=last_date,
            y0=y_min - y_pad, y1=y_max + y_pad,
            line=dict(color='#ef4444', width=1.5, dash='dot'),
            layer='above',
        )

        # ── Layout: dark theme matching the reference image ──────
        ticker_label = f"{selected_ticker}.JK" if not selected_ticker.endswith('.JK') else selected_ticker
        fig.update_layout(
            title=dict(
                text=f"Stock Price Prediction for {ticker_label}",
                font=dict(color='#e2e8f0', size=14, family='Inter'),
                x=0, xanchor='left',
                pad=dict(l=8, t=4),
            ),
            height=420,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#8b949e', family='Inter', size=11),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
                orientation='v',
                yanchor='top', y=0.98,
                xanchor='right', x=0.99,
                font=dict(color='#c9d1d9', size=11),
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='#21262d',
                gridwidth=1,
                zeroline=False,
                tickfont=dict(color='#8b949e', size=10),
                title=dict(text='Date', font=dict(color='#8b949e', size=11)),
                tickformat='%b %-d\n%Y',
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#21262d',
                gridwidth=1,
                zeroline=False,
                tickfont=dict(color='#8b949e', size=10),
                title=dict(text='Price', font=dict(color='#8b949e', size=11)),
                tickformat=',',
            ),
            margin=dict(l=60, r=20, t=48, b=48),
            hovermode='x unified',
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown(f'<div class="section-header">Ringkasan Prediksi</div>', unsafe_allow_html=True)
        
        if model_choice == "XGBoost":
            active_pred = xgb_preds
        elif model_choice == "LSTM":
            active_pred = lstm_preds
        else:
            active_pred = ens_preds
        
        pred_last   = active_pred[-1]
        pred_change = pred_last - hist_prices[-1]
        pred_pct    = pred_change / hist_prices[-1] * 100
        up_cls      = "prediction-up" if pred_change >= 0 else "prediction-down"
        sign        = "+" if pred_change >= 0 else ""
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.8rem">
            <div class="metric-label">Prediksi {future_days} Hari</div>
            <div class="metric-value">Rp {pred_last:,.0f}</div>
            <div class="{up_cls}">{sign}{pred_change:,.0f} ({sign}{pred_pct:.2f}%)</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.8rem">
            <div class="metric-label">Tertinggi Prediksi</div>
            <div class="metric-value prediction-up">Rp {max(active_pred):,.0f}</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.8rem">
            <div class="metric-label">Terendah Prediksi</div>
            <div class="metric-value prediction-down">Rp {min(active_pred):,.0f}</div>
        </div>""", unsafe_allow_html=True)
        
        # Model performance simulation
        st.markdown('<div class="section-header">Performa Model (Simulasi)</div>', unsafe_allow_html=True)
        perf = {
            "XGBoost":  {"RMSE": "245.3", "MAE": "189.2", "R²": "0.94"},
            "LSTM":     {"RMSE": "198.7", "MAE": "161.5", "R²": "0.96"},
            "Ensemble (XGB + LSTM)": {"RMSE": "175.4", "MAE": "142.3", "R²": "0.97"},
        }[model_choice]
        for k, v in perf.items():
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #1e293b'>"
                        f"<span style='color:#94a3b8;font-size:0.85rem'>{k}</span>"
                        f"<span style='color:#f8fafc;font-weight:600;font-size:0.85rem'>{v}</span></div>",
                        unsafe_allow_html=True)

        # Trend signal
        st.markdown('<div class="section-header">Sinyal Teknikal</div>', unsafe_allow_html=True)
        bull = latest['MA5'] > latest['MA20']
        trend_lbl = "Bullish" if bull else "Bearish"
        trend_color = "#10b981" if bull else "#ef4444"
        rsi_proxy = min(max((latest['Close_Price'] - latest['MA20']) / (latest['MA20'] + 1e-9) * 100 + 50, 10), 90)
        rsi_note = "(Overbought)" if rsi_proxy > 70 else "(Oversold)" if rsi_proxy < 30 else "(Netral)"
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:10px;padding:1rem;'>
            <div style='color:{trend_color};font-size:1rem;font-weight:600;margin-bottom:0.5rem'>{trend_lbl}</div>
            <div style='color:#94a3b8;font-size:0.82rem'>MA5 {'>' if bull else '<'} MA20 — {'Uptrend' if bull else 'Downtrend'}</div>
            <div style='color:#94a3b8;font-size:0.82rem;margin-top:0.3rem'>RSI Proxy: {rsi_proxy:.0f} {rsi_note}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Chatbot
# ══════════════════════════════════════════════════════════════
with tab2:
    SYSTEM_PROMPT = """Kamu adalah StockMaster AI, asisten analis saham Indonesia yang cerdas dan membantu.
Kamu memiliki akses ke data fundamental dan teknikal saham-saham IDX (Bursa Efek Indonesia).

Panduan menjawab:
- Jawab berdasarkan DATA KONTEKS yang diberikan. Jangan mengarang angka.
- Jika data tidak tersedia dalam konteks, katakan dengan jujur.
- Gunakan Bahasa Indonesia yang jelas dan mudah dipahami investor pemula.
- Sertakan satuan mata uang (Rupiah) dan tanggal saat menyebutkan angka.
- Berikan penjelasan singkat tentang apa arti metrik (ROE, DER, dll) jika diperlukan.
- JANGAN memberikan rekomendasi beli/jual yang pasti. Sampaikan sebagai analisis informatif.
- Tambahkan disclaimer bahwa analisis bukan saran investasi resmi."""

    col_chat, col_info = st.columns([3, 1])

    with col_chat:
        st.markdown('<div class="section-header">Financial Advisor</div>', unsafe_allow_html=True)

        if not groq_available:
            st.error("Fitur Chatbot tidak aktif karena package groq belum terinstal.")
            st.info("Silakan instal dengan menjalankan perintah berikut di terminal Anda:\n```bash\npip install groq\n```\nSetelah instalasi selesai, jalankan ulang atau refresh halaman Streamlit ini.")

        # Chat display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="chat-bubble-bot">
                    Halo! Saya <b>StockMaster AI</b>, asisten analisis saham IDX Anda.<br><br>
                    Saya bisa membantu Anda dengan:<br>
                    - Analisis harga dan tren saham<br>
                    - Interpretasi indikator teknikal (MA, Volatilitas)<br>
                    - Analisis fundamental (ROE, DER, ROA, EPS)<br>
                    - Perbandingan antar saham<br><br>
                    Silakan tanyakan apapun tentang saham IDX. Pastikan API Key Groq sudah diisi di sidebar.
                </div>""", unsafe_allow_html=True)
            
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-role-user">Anda</div>
                    <div class="chat-bubble-user">{msg['display_content']}</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-role-bot">StockMaster AI</div>
                    <div class="chat-bubble-bot">{msg['content']}</div>
                    """, unsafe_allow_html=True)
        
        # Input
        st.markdown("")
        if groq_available:
            user_input = st.chat_input("Tanya tentang saham IDX... (contoh: Analisis BBCA, Bandingkan ROE saham bank)")
        else:
            user_input = None
            st.chat_input("Chatbot dinonaktifkan (groq belum terinstal)", disabled=True)
        
        if user_input:
            # Build RAG context
            context = build_rag_context(df, selected_ticker, user_input)
            augmented_msg = f"""DATA KONTEKS SAHAM {selected_ticker}:
{context}

PERTANYAAN USER:
{user_input}"""
            
            # Append to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': augmented_msg,
                'display_content': user_input,
            })
            
            # Build messages for API
            api_messages = [
                {'role': m['role'], 'content': m['content']}
                for m in st.session_state.chat_history
            ]
            
            with st.spinner("Menganalisis..."):
                response = call_groq_api(api_messages, SYSTEM_PROMPT)

            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'display_content': response,
            })
            st.rerun()
        
        # Reset button
        if st.session_state.chat_history:
            if st.button("Reset Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    with col_info:
        st.markdown('<div class="section-header">Contoh Pertanyaan</div>', unsafe_allow_html=True)
        
        examples = [
            f"Berapa harga penutupan {selected_ticker} terbaru?",
            f"Bagaimana kondisi fundamental {selected_ticker}?",
            f"Apakah {selected_ticker} sedang uptrend atau downtrend?",
            "Bandingkan ROE semua saham bank",
            f"Jelaskan arti DER dan ROE {selected_ticker}",
            "Saham mana yang paling volatil?",
        ]
        
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True, disabled=not groq_available):
                context = build_rag_context(df, selected_ticker, ex)
                augmented = f"DATA KONTEKS SAHAM {selected_ticker}:\n{context}\n\nPERTANYAAAN USER:\n{ex}"
                st.session_state.chat_history.append({
                    'role': 'user', 'content': augmented, 'display_content': ex,
                })
                api_messages = [{'role': m['role'], 'content': m['content']} for m in st.session_state.chat_history]
                with st.spinner("Menganalisis..."):
                    response = call_groq_api(api_messages, SYSTEM_PROMPT)
                st.session_state.chat_history.append({
                    'role': 'assistant', 'content': response, 'display_content': response,
                })
                st.rerun()
        
        st.markdown('<div class="section-header">Saham Aktif (ROE Tertinggi)</div>', unsafe_allow_html=True)
        df_latest = df.groupby('Ticker').last().reset_index()
        for _, row in df_latest.sort_values('ROE', ascending=False).head(5).iterrows():
            roe_color = "#10b981" if row['ROE'] > 0.15 else "#f59e0b"
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                        display:flex;justify-content:space-between;align-items:center'>
                <span style='color:#e2e8f0;font-weight:600;font-size:0.88rem'>{row['Ticker']}</span>
                <span style='color:{roe_color};font-size:0.82rem'>ROE {row['ROE']*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Data & Fundamental
# ══════════════════════════════════════════════════════════════
with tab3:
    import plotly.express as px
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-header">Data Historis Terbaru</div>', unsafe_allow_html=True)
        df_show = df_sel.sort_values('Tanggal', ascending=False).head(20).copy()
        df_show['Tanggal'] = df_show['Tanggal'].dt.strftime('%d %b %Y')
        df_show['Close_Price'] = df_show['Close_Price'].apply(lambda x: f"Rp {x:,.0f}")
        df_show['ROE'] = df_show['ROE'].apply(lambda x: f"{x*100:.2f}%")
        df_show['ROA'] = df_show['ROA'].apply(lambda x: f"{x*100:.2f}%")
        df_show = df_show[['Tanggal','Close_Price','MA5','MA20','Volatility20','ROE','ROA','DER']].rename(
            columns={'Close_Price':'Harga','Volatility20':'Volatil'}
        )
        st.dataframe(df_show, use_container_width=True, height=340)

    with col_b:
        st.markdown('<div class="section-header">Perbandingan Fundamental</div>', unsafe_allow_html=True)
        df_comp = df.groupby('Ticker').last().reset_index()
        metric_sel = st.selectbox("Pilih Metrik", ["ROE", "ROA", "DER", "Basic EPS", "Volatility20"], index=0)
        
        if metric_sel in ["ROE", "ROA"]:
            df_comp[metric_sel + "_pct"] = df_comp[metric_sel] * 100
            fig_bar = px.bar(
                df_comp.sort_values(metric_sel + "_pct", ascending=False),
                x='Ticker', y=metric_sel + "_pct",
                color=metric_sel + "_pct",
                color_continuous_scale='Blues',
                labels={metric_sel + "_pct": f"{metric_sel} (%)"},
            )
        else:
            fig_bar = px.bar(
                df_comp.sort_values(metric_sel, ascending=False),
                x='Ticker', y=metric_sel,
                color=metric_sel,
                color_continuous_scale='Blues',
            )
        
        fig_bar.update_layout(
            height=320, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.8)',
            font=dict(color='#94a3b8'),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor='#1e293b')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">Heatmap Korelasi Fitur</div>', unsafe_allow_html=True)
    
    features = ['Close_Price', 'Basic EPS', 'DER', 'ROA', 'ROE', 'MA5', 'MA20', 'Volatility20']
    corr = df_sel[features].corr().round(2)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig_heat.update_layout(
        height=380, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        font=dict(color='#94a3b8'),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.76rem;padding:1.5rem 0 0.5rem;
            border-top:1px solid #1e293b;margin-top:1.5rem'>
    <b>Disclaimer:</b> Analisis ini bersifat informatif dan bukan merupakan saran investasi resmi.
    Selalu lakukan riset mendalam dan konsultasikan dengan penasihat keuangan sebelum berinvestasi.
    &nbsp;|&nbsp; StockMaster AI &copy; 2025
</div>
""", unsafe_allow_html=True)
