# ==============================================================================
# IMPORT LIBRARY
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="LSTM StockVision · Prediksi Return Saham",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS — Tampilan Dark Premium
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.main { background-color: #0A0E1A; }
[data-testid="stSidebar"] {
    background-color: #0F1526;
    border-right: 1px solid #1E2A45;
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #0F1526 0%, #141B30 100%);
    border: 1px solid #1E2A45;
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 12px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.accent-blue::before  { background: #00D4FF; }
.kpi-card.accent-green::before { background: #00FF94; }
.kpi-card.accent-purple::before{ background: #7B61FF; }
.kpi-card.accent-orange::before{ background: #FF6B35; }
.kpi-card.accent-yellow::before{ background: #FFB800; }

.kpi-icon  { font-size: 24px; margin-bottom: 6px; }
.kpi-label { font-size: 11px; color: #5A6A85; letter-spacing: 1.5px; text-transform: uppercase; font-family: 'Space Mono', monospace; margin-bottom: 6px; }
.kpi-value { font-size: 26px; font-weight: 800; color: #E8EDF5; font-family: 'Space Mono', monospace; }
.kpi-delta { font-size: 12px; margin-top: 4px; font-family: 'Space Mono', monospace; }
.kpi-delta.up   { color: #00FF94; }
.kpi-delta.down { color: #FF6B35; }
.kpi-delta.neu  { color: #FFB800; }

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E2A45;
}
.section-header h3 {
    font-size: 16px;
    font-weight: 700;
    color: #E8EDF5;
    margin: 0;
    letter-spacing: .3px;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* Signal badges */
.badge-signal {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: .5px;
}
.badge-buy  { background:#00FF9420; color:#00FF94; border:1px solid #00FF9450; }
.badge-sell { background:#FF6B3520; color:#FF6B35; border:1px solid #FF6B3550; }
.badge-hold { background:#FFB80020; color:#FFB800; border:1px solid #FFB80050; }

/* Insight box */
.insight-box {
    background: #0F1526;
    border: 1px solid #1E2A45;
    border-left: 4px solid #00D4FF;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 14px;
    color: #A0B0C8;
    line-height: 1.7;
    margin: 12px 0;
}
.insight-box.green  { border-left-color: #00FF94; }
.insight-box.purple { border-left-color: #7B61FF; }
.insight-box.orange { border-left-color: #FF6B35; }

/* Indicator row */
.indicator-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 10px 0;
}
.indicator-chip {
    background: #141B30;
    border: 1px solid #1E2A45;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
}
.indicator-chip span { font-weight: 700; }
.ic-green span { color: #00FF94; }
.ic-red   span { color: #FF6B35; }
.ic-yellow span{ color: #FFB800; }
.ic-blue  span { color: #00D4FF; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================
@st.cache_data
def load_data():
    df_raw          = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sent    = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_tech    = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sent    = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_tech    = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")
    return df_raw, df_pred_sent, df_pred_tech, df_eval_sent, df_eval_tech

# ------------------------------------------------------------------
# File CSV yang digunakan (sesuai nama di repo GitHub):
#   dataset_merger_covid2020.csv       — dataset historis utama
#   hasil_prediksi_xgb_sentimen_t1.csv — hasil prediksi model sentimen
#   hasil_prediksi_xgb_teknikal_t1.csv — hasil prediksi model teknikal
#   evaluasi_xgb_sentimen_t1.csv       — evaluasi model sentimen
#   evaluasi_xgb_teknikal_t1.csv       — evaluasi model teknikal
#
# Kolom wajib file prediksi: 'Actual' (atau kolom pertama), 'Prediction' (atau kolom terakhir)
# Kolom wajib file evaluasi: 'RMSE' (wajib), 'MAE', 'R2' (opsional)
# ------------------------------------------------------------------

try:
    df_raw, df_pred_sent, df_pred_tech, df_eval_sent, df_eval_tech = load_data()
except Exception as e:
    st.error(f"⚠️ Gagal membaca file CSV: {e}")
    st.info("Pastikan semua file CSV berada satu folder dengan app.py dan nama file sudah sesuai.")
    st.stop()

# ==============================================================================
# HELPER: Hitung Indikator Teknikal dari df_raw
# ==============================================================================
def compute_indicators(df):
    out = df.copy()
    close = out['Close']

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    out['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['MACD']        = ema12 - ema26
    out['MACD_signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
    out['MACD_hist']   = out['MACD'] - out['MACD_signal']

    # Bollinger Bands (20)
    sma20            = close.rolling(20).mean()
    std20            = close.rolling(20).std()
    out['BB_upper']  = sma20 + 2 * std20
    out['BB_lower']  = sma20 - 2 * std20
    out['BB_mid']    = sma20

    # SMA
    out['SMA_20']  = close.rolling(20).mean()
    out['SMA_50']  = close.rolling(50).mean()
    out['SMA_200'] = close.rolling(200).mean()

    # ATR-14
    if 'High' in df.columns and 'Low' in df.columns:
        tr = pd.concat([
            (out['High'] - out['Low']),
            (out['High'] - close.shift()).abs(),
            (out['Low']  - close.shift()).abs()
        ], axis=1).max(axis=1)
        out['ATR'] = tr.rolling(14).mean()

    # Volume MA
    if 'Volume' in df.columns:
        out['Vol_MA20'] = out['Volume'].rolling(20).mean()

    return out

df_tech = compute_indicators(df_raw)

# ==============================================================================
# PREPARASI METRIK
# ==============================================================================
harga_terakhir  = df_raw['Close'].iloc[-1]   if 'Close'  in df_raw.columns else 0
return_terakhir = df_raw['Close'].pct_change().iloc[-1] * 100 if 'Close' in df_raw.columns else 0

def get_metric(df, candidates, default=0):
    for c in candidates:
        if c in df.columns:
            try: return float(df[c].iloc[0])
            except: pass
    return default

rmse_sent   = get_metric(df_eval_sent, ['RMSE','rmse','Root Mean Squared Error','root_mean_squared_error'])
mae_sent    = get_metric(df_eval_sent, ['MAE','mae','Mean Absolute Error','mean_absolute_error'])
r2_sent     = get_metric(df_eval_sent, ['R2','r2','R-Squared','r_squared','R2 Score'])

rmse_tech   = get_metric(df_eval_tech, ['RMSE','rmse','Root Mean Squared Error','root_mean_squared_error'])
mae_tech    = get_metric(df_eval_tech, ['MAE','mae','Mean Absolute Error','mean_absolute_error'])
r2_tech     = get_metric(df_eval_tech, ['R2','r2','R-Squared','r_squared','R2 Score'])

model_terbaik = "LSTM + Sentimen 📰" if rmse_sent < rmse_tech else "LSTM Teknikal 💻"
rmse_terbaik  = min(rmse_sent, rmse_tech)

# Sentimen terkini (kolom 'Sentiment' atau 'sentiment_score')
sent_cols = [c for c in df_raw.columns if 'sent' in c.lower()]
sent_score = df_raw[sent_cols[0]].iloc[-1] if sent_cols else None

# Sinyal teknikal terakhir
last = df_tech.iloc[-1]
rsi_val   = last.get('RSI', np.nan)
macd_val  = last.get('MACD', np.nan)
macd_sig  = last.get('MACD_signal', np.nan)

def sinyal_teknikal(rsi, macd, macd_signal):
    skor = 0
    if not np.isnan(rsi):
        if rsi < 30: skor += 2
        elif rsi < 50: skor += 1
        elif rsi > 70: skor -= 2
        else: skor -= 1
    if not np.isnan(macd) and not np.isnan(macd_signal):
        skor += 1 if macd > macd_signal else -1
    if skor >= 2: return "BUY", "badge-buy"
    if skor <= -2: return "SELL", "badge-sell"
    return "HOLD", "badge-hold"

signal_label, signal_class = sinyal_teknikal(rsi_val, macd_val, macd_sig)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 10px'>
      <div style='font-size:32px'>⚡</div>
      <div style='font-size:18px;font-weight:800;letter-spacing:-.5px'>LSTM StockVision</div>
      <div style='font-size:10px;color:#5A6A85;letter-spacing:2px;margin-top:4px;font-family:Space Mono,monospace'>DEEP LEARNING · SAHAM ENERGI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigasi:",
        ["🏠 Dashboard", "📈 Prediksi LSTM", "📊 Analisis Teknikal", "💬 Analisis Sentimen", "📋 Evaluasi Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#5A6A85;line-height:1.8;padding:0 4px'>
    <b style='color:#A0B0C8'>Model:</b> LSTM (Long Short-Term Memory)<br>
    <b style='color:#A0B0C8'>Dataset:</b> Saham Energi LQ45<br>
    <b style='color:#A0B0C8'>Periode:</b> Pandemi COVID-19<br>
    <b style='color:#A0B0C8'>Framework:</b> TensorFlow / Keras<br>
    <b style='color:#A0B0C8'>Fitur:</b> Teknikal + Sentimen NLP
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("ℹ️ Tentang LSTM"):
        st.markdown("""
        **Long Short-Term Memory (LSTM)** adalah arsitektur *recurrent neural network* yang dirancang khusus untuk memodelkan ketergantungan jangka panjang dalam data sekuensial seperti harga saham.
        
        Model ini mengatasi masalah *vanishing gradient* yang umum pada RNN konvensional melalui mekanisme *forget gate*, *input gate*, dan *output gate*.
        """)

# ==============================================================================
# ─── HALAMAN 1: DASHBOARD ───────────────────────────────────────────────────
# ==============================================================================
if menu == "🏠 Dashboard":

    st.markdown("""
    <div style='padding:10px 0 4px'>
      <h1 style='font-size:28px;font-weight:800;margin:0;color:#E8EDF5'>
        ⚡ Dashboard Prediksi Return Saham
      </h1>
      <p style='font-size:14px;color:#5A6A85;margin:6px 0 0;font-family:Space Mono,monospace'>
        LSTM · Saham Sektor Energi LQ45 · Periode Pandemi COVID-19
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI Row 1
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="kpi-card accent-blue">
          <div class="kpi-icon">💰</div>
          <div class="kpi-label">Harga Terakhir</div>
          <div class="kpi-value">Rp {harga_terakhir:,.0f}</div>
          <div class="kpi-delta {'up' if return_terakhir >= 0 else 'down'}">
            {'▲' if return_terakhir >= 0 else '▼'} {abs(return_terakhir):.2f}%
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card accent-green">
          <div class="kpi-icon">🧠</div>
          <div class="kpi-label">RMSE Terbaik</div>
          <div class="kpi-value">{rmse_terbaik:.5f}</div>
          <div class="kpi-delta neu">MODEL LSTM</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card accent-purple">
          <div class="kpi-icon">📰</div>
          <div class="kpi-label">RMSE Sentimen</div>
          <div class="kpi-value">{rmse_sent:.5f}</div>
          <div class="kpi-delta neu">R² = {r2_sent:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card accent-orange">
          <div class="kpi-icon">📊</div>
          <div class="kpi-label">RMSE Teknikal</div>
          <div class="kpi-value">{rmse_tech:.5f}</div>
          <div class="kpi-delta neu">R² = {r2_tech:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        sig_color = {"BUY": "up", "SELL": "down", "HOLD": "neu"}[signal_label]
        sig_icon  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal_label]
        st.markdown(f"""
        <div class="kpi-card accent-yellow">
          <div class="kpi-icon">{sig_icon}</div>
          <div class="kpi-label">Sinyal Teknikal</div>
          <div class="kpi-value">{signal_label}</div>
          <div class="kpi-delta neu">RSI: {rsi_val:.1f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Insight
    st.markdown(f"""
    <div class="insight-box green">
      🏆 <b>Model Terbaik:</b> {model_terbaik} &nbsp;|&nbsp; RMSE: <b>{rmse_terbaik:.5f}</b><br>
      Berdasarkan nilai galat terkecil dari dua pendekatan LSTM (Teknikal vs Sentimen), model dengan performa optimal adalah <b>{model_terbaik}</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Mini Price Chart
    st.markdown("""<div class="section-header">
      <div class="section-dot" style="background:#00D4FF"></div>
      <h3>Harga Penutupan Historis</h3>
    </div>""", unsafe_allow_html=True)

    if 'Close' in df_raw.columns:
        close_series = df_raw['Close'].reset_index(drop=True)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            y=close_series, mode='lines',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.06)',
            name='Close'
        ))
        fig_price.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,20,36,0.6)',
            height=260,
            margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(showgrid=False, color='#5A6A85'),
            yaxis=dict(gridcolor='#1E2A45', color='#5A6A85'),
            showlegend=False
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Ringkasan penelitian
    st.markdown("""<div class="section-header">
      <div class="section-dot" style="background:#7B61FF"></div>
      <h3>Ringkasan Penelitian</h3>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="insight-box purple">
          <b>🔬 Metode LSTM (Deep Learning)</b><br><br>
          Penelitian ini menggunakan arsitektur <b>LSTM (Long Short-Term Memory)</b> — 
          jaringan saraf berulang yang sangat efektif untuk data time-series keuangan. 
          LSTM mampu "mengingat" pola jangka panjang dari pergerakan harga saham 
          untuk menghasilkan prediksi return yang lebih akurat dibandingkan model klasik.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="insight-box orange">
          <b>📊 Dua Pendekatan Fitur</b><br><br>
          <b>1. Analisis Teknikal:</b> Menggunakan indikator pasar seperti RSI, MACD, 
          Bollinger Bands, SMA, dan Volume sebagai fitur input LSTM.<br><br>
          <b>2. Analisis Sentimen:</b> Mengekstrak skor sentimen dari berita/media 
          menggunakan NLP, dikombinasikan dengan data harga sebagai fitur LSTM.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# ─── HALAMAN 2: PREDIKSI LSTM ────────────────────────────────────────────────
# ==============================================================================
elif menu == "📈 Prediksi LSTM":

    st.markdown("""
    <h2 style='font-size:22px;font-weight:800;color:#E8EDF5;margin-bottom:4px'>
      📈 Visualisasi Prediksi Model LSTM
    </h2>
    <p style='color:#5A6A85;font-size:13px;font-family:Space Mono,monospace;margin-bottom:16px'>
      Perbandingan nilai aktual vs prediksi — Model Sentimen & Model Teknikal
    </p>
    """, unsafe_allow_html=True)

    model_pilih = st.selectbox(
        "Tampilkan model:",
        ["Keduanya (Perbandingan)", "LSTM + Sentimen", "LSTM Teknikal"],
        label_visibility="visible"
    )

    # Susun dataframe plot — auto-detect nama kolom apapun
    n = min(len(df_pred_sent), len(df_pred_tech))
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        return df.columns[0]
    actual_col = find_col(df_pred_sent, ['Actual','actual','y_test','y_true','Close','close','Return','return'])
    pred_s_col = find_col(df_pred_sent, ['Prediction','prediction','y_pred','Predicted','predicted','y_pred_sent'])
    pred_t_col = find_col(df_pred_tech,  ['Prediction','prediction','y_pred','Predicted','predicted','y_pred_tech'])

    aktual    = df_pred_sent[actual_col].values[:n]
    pred_sent = df_pred_sent[pred_s_col].values[:n]
    pred_tech = df_pred_tech[pred_t_col].values[:n]
    x_idx     = list(range(n))

    fig2 = go.Figure()

    # Aktual
    fig2.add_trace(go.Scatter(
        x=x_idx, y=aktual, mode='lines',
        name='Aktual',
        line=dict(color='#00D4FF', width=2.5),
    ))

    if model_pilih in ["Keduanya (Perbandingan)", "LSTM + Sentimen"]:
        fig2.add_trace(go.Scatter(
            x=x_idx, y=pred_sent, mode='lines',
            name='LSTM + Sentimen',
            line=dict(color='#7B61FF', width=2, dash='dash'),
        ))

    if model_pilih in ["Keduanya (Perbandingan)", "LSTM Teknikal"]:
        fig2.add_trace(go.Scatter(
            x=x_idx, y=pred_tech, mode='lines',
            name='LSTM Teknikal',
            line=dict(color='#FF6B35', width=2, dash='dot'),
        ))

    # Error band sentimen
    if model_pilih in ["Keduanya (Perbandingan)", "LSTM + Sentimen"]:
        fig2.add_trace(go.Scatter(
            x=x_idx + x_idx[::-1],
            y=list(pred_sent * 1.02) + list(pred_sent[::-1] * 0.98),
            fill='toself', fillcolor='rgba(123,97,255,0.07)',
            line=dict(color='rgba(255,255,255,0)'),
            name='CI Sentimen', showlegend=False
        ))

    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14,20,36,0.6)',
        height=480,
        title=dict(
            text="<b>Aktual vs Prediksi LSTM</b>",
            font=dict(size=16, color='#E8EDF5'), x=0
        ),
        xaxis=dict(title='Index Pengujian', gridcolor='#1E2A45', color='#5A6A85'),
        yaxis=dict(title='Nilai / Harga', gridcolor='#1E2A45', color='#5A6A85'),
        legend=dict(
            bgcolor='rgba(15,21,38,0.9)',
            bordercolor='#1E2A45', borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Error distribution
    st.markdown("""<div class="section-header">
      <div class="section-dot" style="background:#00FF94"></div>
      <h3>Distribusi Error Prediksi</h3>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        err_s = pred_sent - aktual
        fig_es = go.Figure(go.Histogram(
            x=err_s, nbinsx=30,
            marker_color='#7B61FF', opacity=0.8, name='Error Sentimen'
        ))
        fig_es.add_vline(x=0, line_dash='dash', line_color='#00FF94', line_width=1.5)
        fig_es.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,20,36,0.6)', height=260,
            title=dict(text='Distribusi Error · Sentimen', font=dict(size=13), x=0),
            margin=dict(l=30, r=10, t=40, b=30),
            xaxis=dict(gridcolor='#1E2A45'), yaxis=dict(gridcolor='#1E2A45'),
            showlegend=False
        )
        st.plotly_chart(fig_es, use_container_width=True)

    with col2:
        err_t = pred_tech - aktual
        fig_et = go.Figure(go.Histogram(
            x=err_t, nbinsx=30,
            marker_color='#FF6B35', opacity=0.8, name='Error Teknikal'
        ))
        fig_et.add_vline(x=0, line_dash='dash', line_color='#00FF94', line_width=1.5)
        fig_et.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,20,36,0.6)', height=260,
            title=dict(text='Distribusi Error · Teknikal', font=dict(size=13), x=0),
            margin=dict(l=30, r=10, t=40, b=30),
            xaxis=dict(gridcolor='#1E2A45'), yaxis=dict(gridcolor='#1E2A45'),
            showlegend=False
        )
        st.plotly_chart(fig_et, use_container_width=True)

    st.markdown("""
    <div class="insight-box" style="margin-top:8px">
      💡 <b>Petunjuk:</b> Kursor hover untuk detail angka. Klik legenda untuk sembunyikan/tampilkan garis.
      Zoom in dengan klik-drag. Double-click untuk reset view.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# ─── HALAMAN 3: ANALISIS TEKNIKAL ────────────────────────────────────────────
# ==============================================================================
elif menu == "📊 Analisis Teknikal":

    st.markdown("""
    <h2 style='font-size:22px;font-weight:800;color:#E8EDF5;margin-bottom:4px'>
      📊 Analisis Indikator Teknikal
    </h2>
    <p style='color:#5A6A85;font-size:13px;font-family:Space Mono,monospace;margin-bottom:16px'>
      RSI · MACD · Bollinger Bands · SMA · Volume
    </p>
    """, unsafe_allow_html=True)

    # Indikator chips
    rsi_now  = df_tech['RSI'].iloc[-1]   if 'RSI'  in df_tech.columns else np.nan
    macd_now = df_tech['MACD'].iloc[-1]  if 'MACD' in df_tech.columns else np.nan
    msig_now = df_tech['MACD_signal'].iloc[-1] if 'MACD_signal' in df_tech.columns else np.nan
    sma20    = df_tech['SMA_20'].iloc[-1] if 'SMA_20' in df_tech.columns else np.nan
    sma50    = df_tech['SMA_50'].iloc[-1] if 'SMA_50' in df_tech.columns else np.nan
    atr_now  = df_tech['ATR'].iloc[-1]   if 'ATR'  in df_tech.columns else np.nan

    rsi_cls  = "ic-green" if rsi_now < 50 else "ic-red"
    macd_cls = "ic-green" if (not np.isnan(macd_now) and not np.isnan(msig_now) and macd_now > msig_now) else "ic-red"
    sma_cls  = "ic-green" if (not np.isnan(sma20) and not np.isnan(sma50) and sma20 > sma50) else "ic-red"

    chips = [
        (rsi_cls,   f"RSI(14): <span>{rsi_now:.1f}</span>"),
        (macd_cls,  f"MACD: <span>{'Bullish ▲' if macd_cls=='ic-green' else 'Bearish ▼'}</span>"),
        (sma_cls,   f"SMA Cross: <span>{'Golden ▲' if sma_cls=='ic-green' else 'Death ▼'}</span>"),
        ("ic-blue",  f"ATR(14): <span>{atr_now:.1f if not np.isnan(atr_now) else 'N/A'}</span>"),
        ("ic-yellow", f"Sinyal: <span>{signal_label}</span>"),
    ]
    chip_html = '<div class="indicator-row">' + ''.join(
        f'<div class="indicator-chip {c}">{t}</div>' for c, t in chips
    ) + '</div>'
    st.markdown(chip_html, unsafe_allow_html=True)

    # ── Candlestick + BB + SMA
    st.markdown("""<div class="section-header">
      <div class="section-dot" style="background:#00D4FF"></div>
      <h3>Harga & Bollinger Bands</h3>
    </div>""", unsafe_allow_html=True)

    df_plot = df_tech.tail(120).reset_index(drop=True)
    fig_c = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          row_heights=[0.55, 0.25, 0.2],
                          vertical_spacing=0.04)

    # Harga + BB
    if all(c in df_plot.columns for c in ['Open','High','Low','Close']):
        fig_c.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'],  close=df_plot['Close'],
            increasing_line_color='#00FF94', decreasing_line_color='#FF6B35',
            name='OHLC'
        ), row=1, col=1)
    else:
        fig_c.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Close'], mode='lines',
            line=dict(color='#00D4FF', width=2), name='Close'
        ), row=1, col=1)

    for col_name, clr, dash, nm in [
        ('BB_upper','#5A6A85','dot','BB Upper'),
        ('BB_mid','#FFB800','dash','BB Mid'),
        ('BB_lower','#5A6A85','dot','BB Lower'),
        ('SMA_20','#00D4FF','solid','SMA 20'),
        ('SMA_50','#7B61FF','dash','SMA 50'),
    ]:
        if col_name in df_plot.columns:
            fig_c.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[col_name], mode='lines',
                line=dict(color=clr, width=1.2, dash=dash), name=nm
            ), row=1, col=1)

    # RSI
    if 'RSI' in df_plot.columns:
        fig_c.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['RSI'], mode='lines',
            line=dict(color='#FFB800', width=1.5), name='RSI'
        ), row=2, col=1)
        fig_c.add_hline(y=70, line_dash='dash', line_color='#FF6B35', line_width=1, row=2, col=1)
        fig_c.add_hline(y=30, line_dash='dash', line_color='#00FF94', line_width=1, row=2, col=1)

    # MACD
    if 'MACD' in df_plot.columns:
        macd_colors = ['#00FF94' if v >= 0 else '#FF6B35' for v in df_plot['MACD_hist'].fillna(0)]
        fig_c.add_trace(go.Bar(
            x=df_plot.index, y=df_plot['MACD_hist'],
            marker_color=macd_colors, name='MACD Hist'
        ), row=3, col=1)
        fig_c.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['MACD'], mode='lines',
            line=dict(color='#7B61FF', width=1.2), name='MACD'
        ), row=3, col=1)
        fig_c.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['MACD_signal'], mode='lines',
            line=dict(color='#FF6B35', width=1.2, dash='dash'), name='Signal'
        ), row=3, col=1)

    fig_c.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14,20,36,0.6)',
        height=680,
        margin=dict(l=40, r=20, t=20, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='rgba(15,21,38,0.9)', bordercolor='#1E2A45',
                    borderwidth=1, font=dict(size=11)),
        yaxis=dict(gridcolor='#1E2A45', color='#5A6A85'),
        yaxis2=dict(gridcolor='#1E2A45', color='#5A6A85', title='RSI'),
        yaxis3=dict(gridcolor='#1E2A45', color='#5A6A85', title='MACD'),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # Volume chart
    if 'Volume' in df_plot.columns:
        st.markdown("""<div class="section-header">
          <div class="section-dot" style="background:#00FF94"></div>
          <h3>Volume Perdagangan</h3>
        </div>""", unsafe_allow_html=True)

        vol_colors = ['#00FF94' if df_plot['Close'].iloc[i] >= df_plot['Close'].iloc[i-1]
                      else '#FF6B35' for i in range(len(df_plot))]
        fig_vol = go.Figure(go.Bar(
            x=df_plot.index, y=df_plot['Volume'],
            marker_color=vol_colors, name='Volume'
        ))
        if 'Vol_MA20' in df_plot.columns:
            fig_vol.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot['Vol_MA20'], mode='lines',
                line=dict(color='#FFB800', width=1.5, dash='dash'), name='MA20 Vol'
            ))
        fig_vol.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,20,36,0.6)', height=220,
            margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(gridcolor='#1E2A45'), yaxis=dict(gridcolor='#1E2A45'),
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig_vol, use_container_width=True)


# ==============================================================================
# ─── HALAMAN 4: ANALISIS SENTIMEN ────────────────────────────────────────────
# ==============================================================================
elif menu == "💬 Analisis Sentimen":

    st.markdown("""
    <h2 style='font-size:22px;font-weight:800;color:#E8EDF5;margin-bottom:4px'>
      💬 Analisis Sentimen
    </h2>
    <p style='color:#5A6A85;font-size:13px;font-family:Space Mono,monospace;margin-bottom:16px'>
      Korelasi sentimen berita/media dengan return saham
    </p>
    """, unsafe_allow_html=True)

    sent_cols_all = [c for c in df_raw.columns if any(k in c.lower() for k in ['sent', 'opini', 'news', 'berita'])]

    if not sent_cols_all:
        st.markdown("""
        <div class="insight-box orange">
          ⚠️ Kolom sentimen tidak ditemukan di dataset. Pastikan dataset memiliki kolom dengan nama 
          mengandung kata 'sentiment', 'sent', 'news', atau 'berita'.
        </div>
        """, unsafe_allow_html=True)

        if 'Close' in df_raw.columns:
            st.markdown("""<div class="section-header">
              <div class="section-dot" style="background:#7B61FF"></div>
              <h3>Distribusi Return Harian</h3>
            </div>""", unsafe_allow_html=True)

            returns = df_raw['Close'].pct_change().dropna() * 100
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=returns, nbinsx=50,
                marker_color='#7B61FF', opacity=0.8
            ))
            fig_ret.add_vline(x=0, line_dash='dash', line_color='#00FF94', line_width=1.5)
            fig_ret.add_vline(x=returns.mean(), line_dash='dash', line_color='#FFB800',
                              line_width=1.5, annotation_text=f'Mean: {returns.mean():.2f}%',
                              annotation_font_color='#FFB800')
            fig_ret.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,20,36,0.6)', height=340,
                title=dict(text='Distribusi Return Harian (%)', font=dict(size=14), x=0),
                xaxis=dict(title='Return (%)', gridcolor='#1E2A45'),
                yaxis=dict(title='Frekuensi', gridcolor='#1E2A45'),
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_ret, use_container_width=True)
    else:
        sent_col = st.selectbox("Pilih kolom sentimen:", sent_cols_all)
        sent_data = df_raw[sent_col].dropna()

        col1, col2 = st.columns(2)
        with col1:
            fig_sh = go.Figure(go.Histogram(
                x=sent_data, nbinsx=40,
                marker_color='#7B61FF', opacity=0.85
            ))
            fig_sh.add_vline(x=sent_data.mean(), line_dash='dash',
                             line_color='#00FF94', line_width=1.5)
            fig_sh.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,20,36,0.6)', height=300,
                title=dict(text='Distribusi Skor Sentimen', font=dict(size=13), x=0),
                margin=dict(l=30, r=10, t=40, b=30),
                xaxis=dict(gridcolor='#1E2A45'), yaxis=dict(gridcolor='#1E2A45')
            )
            st.plotly_chart(fig_sh, use_container_width=True)

        with col2:
            if 'Close' in df_raw.columns:
                returns2 = df_raw['Close'].pct_change() * 100
                df_scatter = pd.DataFrame({'Sentimen': sent_data, 'Return': returns2}).dropna()
                fig_sc = go.Figure(go.Scatter(
                    x=df_scatter['Sentimen'], y=df_scatter['Return'],
                    mode='markers',
                    marker=dict(color=df_scatter['Return'],
                                colorscale=[[0,'#FF6B35'],[0.5,'#FFB800'],[1,'#00FF94']],
                                size=5, opacity=0.7)
                ))
                m, b = np.polyfit(df_scatter['Sentimen'], df_scatter['Return'], 1)
                x_line = np.linspace(df_scatter['Sentimen'].min(), df_scatter['Sentimen'].max(), 50)
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=m * x_line + b,
                    mode='lines', line=dict(color='#00D4FF', width=2, dash='dash'),
                    name=f'Trend (r={df_scatter.corr().iloc[0,1]:.3f})'
                ))
                fig_sc.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(14,20,36,0.6)', height=300,
                    title=dict(text='Sentimen vs Return', font=dict(size=13), x=0),
                    margin=dict(l=30, r=10, t=40, b=30),
                    xaxis=dict(title='Sentimen', gridcolor='#1E2A45'),
                    yaxis=dict(title='Return (%)', gridcolor='#1E2A45')
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        # Time series sentimen
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            y=sent_data.reset_index(drop=True), mode='lines',
            line=dict(color='#7B61FF', width=1.5),
            fill='tozeroy', fillcolor='rgba(123,97,255,0.07)',
            name='Sentimen'
        ))
        fig_ts.add_hline(y=0, line_color='#5A6A85', line_width=0.8)
        fig_ts.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,20,36,0.6)', height=260,
            title=dict(text='Time Series Skor Sentimen', font=dict(size=13), x=0),
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis=dict(gridcolor='#1E2A45'), yaxis=dict(gridcolor='#1E2A45')
        )
        st.plotly_chart(fig_ts, use_container_width=True)


# ==============================================================================
# ─── HALAMAN 5: EVALUASI MODEL ───────────────────────────────────────────────
# ==============================================================================
elif menu == "📋 Evaluasi Model":

    st.markdown("""
    <h2 style='font-size:22px;font-weight:800;color:#E8EDF5;margin-bottom:4px'>
      📋 Evaluasi & Perbandingan Model LSTM
    </h2>
    <p style='color:#5A6A85;font-size:13px;font-family:Space Mono,monospace;margin-bottom:16px'>
      Matriks evaluasi · Dataset historis · Download data
    </p>
    """, unsafe_allow_html=True)

    # ── Kartu evaluasi
    metrics_list = []
    metric_map = {
        'RMSE': ['RMSE','rmse','Root Mean Squared Error'],
        'MAE':  ['MAE','mae','Mean Absolute Error'],
        'R2':   ['R2','r2','R-Squared','R2 Score'],
        'MAPE': ['MAPE','mape','Mean Absolute Percentage Error'],
    }
    for display_name, candidates in metric_map.items():
        vs, vt = None, None
        for c in candidates:
            if c in df_eval_sent.columns:
                try: vs = float(df_eval_sent[c].iloc[0])
                except: pass
                break
        for c in candidates:
            if c in df_eval_tech.columns:
                try: vt = float(df_eval_tech[c].iloc[0])
                except: pass
                break
        if vs is not None or vt is not None:
            metrics_list.append((display_name, vs, vt))

    if metrics_list:
        cols_m = st.columns(len(metrics_list))
        for idx, (name, vs, vt) in enumerate(metrics_list):
            with cols_m[idx]:
                better = "Sentimen" if (vs is not None and vt is not None and
                                        (vs < vt if name != 'R2' else vs > vt)) else "Teknikal"
                fmt = lambda v: f"{v:.5f}" if v is not None else "N/A"
                st.markdown(f"""
                <div class="kpi-card accent-purple">
                  <div class="kpi-label">{name}</div>
                  <div style='font-size:13px;color:#A0B0C8;margin:6px 0 2px;font-family:Space Mono,monospace'>
                    📰 Sentimen: <b style='color:#7B61FF'>{fmt(vs)}</b>
                  </div>
                  <div style='font-size:13px;color:#A0B0C8;font-family:Space Mono,monospace'>
                    📊 Teknikal: <b style='color:#FF6B35'>{fmt(vt)}</b>
                  </div>
                  <div class='kpi-delta neu' style='margin-top:6px'>✓ {better}</div>
                </div>""", unsafe_allow_html=True)

    # ── Radar chart perbandingan
    if len(metrics_list) >= 3:
        st.markdown("""<div class="section-header">
          <div class="section-dot" style="background:#7B61FF"></div>
          <h3>Radar Chart Perbandingan Model</h3>
        </div>""", unsafe_allow_html=True)

        cats  = [m[0] for m in metrics_list]
        vals_s = [abs(m[1]) if m[1] is not None else 0 for m in metrics_list]
        vals_t = [abs(m[2]) if m[2] is not None else 0 for m in metrics_list]
        mx = [max(a, b) + 1e-9 for a, b in zip(vals_s, vals_t)]
        vals_s_n = [v/m for v,m in zip(vals_s, mx)]
        vals_t_n = [v/m for v,m in zip(vals_t, mx)]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_s_n + [vals_s_n[0]], theta=cats + [cats[0]],
            fill='toself', fillcolor='rgba(123,97,255,0.15)',
            line=dict(color='#7B61FF', width=2), name='LSTM Sentimen'
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=vals_t_n + [vals_t_n[0]], theta=cats + [cats[0]],
            fill='toself', fillcolor='rgba(255,107,53,0.15)',
            line=dict(color='#FF6B35', width=2), name='LSTM Teknikal'
        ))
        fig_r.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                bgcolor='rgba(14,20,36,0.8)',
                radialaxis=dict(gridcolor='#1E2A45', color='#5A6A85'),
                angularaxis=dict(gridcolor='#1E2A45', color='#A0B0C8')
            ),
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(bgcolor='rgba(15,21,38,0.9)', bordercolor='#1E2A45', borderwidth=1)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Tabs data
    st.markdown("""<div class="section-header">
      <div class="section-dot" style="background:#00D4FF"></div>
      <h3>Eksplorasi Data</h3>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 Dataset Historis", "📰 Prediksi Sentimen", "📊 Prediksi Teknikal"])

    with tab1:
        st.dataframe(
            df_raw.tail(50).style.background_gradient(subset=['Close'] if 'Close' in df_raw.columns else [],
                                                       cmap='Blues'),
            use_container_width=True
        )
        st.download_button(
            "📥 Download Dataset (CSV)",
            data=df_raw.to_csv(index=False),
            file_name="dataset_covid_energi.csv", mime="text/csv"
        )

    with tab2:
        col_t, col_e = st.columns([2, 1])
        with col_t:
            st.caption("**Sampel 30 baris prediksi — Model LSTM Sentimen**")
            st.dataframe(df_pred_sent.head(30), use_container_width=True)
        with col_e:
            st.caption("**Matriks evaluasi**")
            st.dataframe(df_eval_sent, use_container_width=True)
        st.download_button(
            "📥 Download Hasil Prediksi Sentimen",
            data=df_pred_sent.to_csv(index=False),
            file_name="prediksi_lstm_sentimen.csv", mime="text/csv"
        )

    with tab3:
        col_t2, col_e2 = st.columns([2, 1])
        with col_t2:
            st.caption("**Sampel 30 baris prediksi — Model LSTM Teknikal**")
            st.dataframe(df_pred_tech.head(30), use_container_width=True)
        with col_e2:
            st.caption("**Matriks evaluasi**")
            st.dataframe(df_eval_tech, use_container_width=True)
        st.download_button(
            "📥 Download Hasil Prediksi Teknikal",
            data=df_pred_tech.to_csv(index=False),
            file_name="prediksi_lstm_teknikal.csv", mime="text/csv"
        )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center;
            padding:8px 0;flex-wrap:wrap;gap:8px'>
  <div style='font-size:12px;color:#5A6A85;font-family:Space Mono,monospace'>
    ⚡ LSTM StockVision · Pemodelan Return Saham Sektor Energi LQ45
  </div>
  <div style='font-size:12px;color:#5A6A85;font-family:Space Mono,monospace'>
    Deep Learning · TensorFlow/Keras · © 2026 Academic Research
  </div>
</div>
""", unsafe_allow_html=True)
