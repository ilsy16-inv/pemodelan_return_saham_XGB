import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Prediksi Volatilitas Saham Energi",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    # Membaca dari susunan folder baru Anda
    df_raw = pd.read_csv("data/raw/dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("data/processed/hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("data/processed/hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("data/processed/evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("data/processed/evaluasi_xgb_teknikal_t1.csv")
    
    # Memastikan format tanggal sesuai
    df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
    df_pred_sentimen['Tanggal'] = pd.to_datetime(df_pred_sentimen['Tanggal'])
    df_pred_teknikal['Tanggal'] = pd.to_datetime(df_pred_teknikal['Tanggal'])
    
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal

# Load data ke aplikasi
df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal = load_data()

# ==========================================
# 3. SIDEBAR (FILTER INTERAKTIF)
# ==========================================
st.sidebar.header("📊 Filter Analisis")

# Filter Pilih Emiten
list_emiten = df_raw['Kode_Emiten'].unique()
pilihan_emiten = st.sidebar.selectbox("Pilih Kode Emiten:", list_emiten)

# Filter Rentang Tanggal
tanggal_min = df_pred_sentimen['Tanggal'].min().to_pydatetime()
tanggal_max = df_pred_sentimen['Tanggal'].max().to_pydatetime()
pilihan_tanggal = st.sidebar.date_input(
    "Rentang Waktu Prediksi:",
    value=[tanggal_min, tanggal_max],
    min_value=tanggal_min,
    max_value=tanggal_max
)

# Filter Data Berdasarkan Pilihan
if isinstance(pilihan_tanggal, list) or isinstance(pilihan_tanggal, tuple):
    if len(pilihan_tanggal) == 2:
        start_date, end_date = pd.to_datetime(pilihan_tanggal[0]), pd.to_datetime(pilihan_tanggal[1])
    else:
        start_date, end_date = pd.to_datetime(tanggal_min), pd.to_datetime(tanggal_max)
else:
    start_date, end_date = pd.to_datetime(pilihan_tanggal), pd.to_datetime(pilihan_tanggal)

sub_raw = df_raw[(df_raw['Kode_Emiten'] == pilihan_emiten) & (df_raw['Tanggal'].between(start_date, end_date))]
sub_pred_s = df_pred_sentimen[(df_pred_sentimen['Kode_Emiten'] == pilihan_emiten) & (df_pred_sentimen['Tanggal'].between(start_date, end_date))]
sub_pred_t = df_pred_teknikal[(df_pred_teknikal['Kode_Emiten'] == pilihan_emiten) & (df_pred_teknikal['Tanggal'].between(start_date, end_date))]

# ==========================================
# 4. KONTEN UTAMA DASHBOARD
# ==========================================
st.title("📈 Dashboard Prediksi Volatilitas Return Saham Sektor Energi (LQ45)")
st.markdown("### Analisis Komparatif XGBoost Berdasarkan Indikator Teknikal & Sentimen Berita")
st.markdown("---")

# --- KARTU INFORMASI UTAMA (KPI) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    harga_terakhir = sub_raw['Close'].iloc[-1] if not sub_raw.empty else 0
    st.metric(label="Harga Penutupan Terakhir", value=f"Rp {harga_terakhir:,.0f}")
with col2:
    sentimen_terakhir = sub_raw['Skor_Sentimen'].iloc[-1] if not sub_raw.empty else 0
    st.metric(label="Skor Sentimen (Hari Ini)", value=f"{sentimen_terakhir:.4f}")
with col3:
    pred_t_terakhir = sub_pred_t['Volatilitas_Pred_XGB_Teknikal_t1'].iloc[-1] if not sub_pred_t.empty else 0
    st.metric(label="Prediksi Volatilitas Besok (Teknikal)", value=f"{pred_t_terakhir*100:.3f}%")
with col4:
    pred_s_terakhir = sub_pred_s['Volatilitas_Pred_XGB_Sentimen_t1'].iloc[-1] if not sub_pred_s.empty else 0
    st.metric(label="Prediksi Volatilitas Besok (Teknikal+Sentimen)", value=f"{pred_s_terakhir*100:.3f}%")

st.markdown("---")

# --- PEMBAGIAN TAB DASHBOARD ---
tab1, tab2 = st.tabs(["📊 Prediksi & Pergerakan Pasar", "🔬 Evaluasi Akurasi Model (Tesis)"])

with tab1:
    # Grafik Harga Historis
    st.subheader("1. Tren Pergerakan Harga Saham Historis")
    fig_harga = go.Figure()
    if not sub_raw.empty:
        fig_harga.add_trace(go.Scatter(x=sub_raw['Tanggal'], y=sub_raw['Close'], name="Harga Close", line=dict(color="#1f77b4", width=2)))
    fig_harga.update_layout(xaxis_title="Tanggal", yaxis_title="Harga (Rp)", height=300, template="plotly_white")
    st.plotly_chart(fig_harga, use_container_width=True)
    
    # Grafik Hasil Forecast Volatilitas
    st.subheader("2. Hasil Forecasting Volatilitas Return Saham ($t+1$)")
    fig_pred = go.Figure()
    if not sub_pred_s.empty:
        fig_pred.add_trace(go.Scatter(x=sub_pred_s['Tanggal'], y=sub_pred_s['Volatilitas_Aktual'], name="Volatilitas Aktual", line=dict(color="black", width=2)))
        fig_pred.add_trace(go.Scatter(x=sub_pred_s['Tanggal'], y=sub_pred_s['Volatilitas_Pred_XGB_Sentimen_t1'], name="Prediksi XGBoost (Teknikal + Sentimen)", line=dict(color="#2ca02c", width=2)))
    if not sub_pred_t.empty:
        fig_pred.add_trace(go.Scatter(x=sub_pred_t['Tanggal'], y=sub_pred_t['Volatilitas_Pred_XGB_Teknikal_t1'], name="Prediksi XGBoost (Teknikal)", line=dict(color="#ff7f0e", dash="dash")))
    
    fig_pred.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Nilai Volatilitas",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with tab2:
    st.subheader(f"Analisis Performa Model Peramalan untuk Emiten {pilihan_emiten}")
    
    eval_t = df_eval_teknikal[df_eval_teknikal['Emiten'] == pilihan_emiten]
    eval_s = df_eval_sentimen[df_eval_sentimen['Emiten'] == pilihan_emiten]
    df_komparasi = pd.concat([eval_t, eval_s], ignore_index=True)
    
    # Menampilkan Tabel Evaluasi
    st.dataframe(df_komparasi.style.format({
        'RMSE': '{:.6f}',
        'MAE': '{:.6f}',
        'MAPE': '{:.2f}%'
    }), use_container_width=True)
    
    # Visualisasi Bar Chart Perbandingan MAPE
    st.markdown("### Perbandingan Tingkat Kesalahan (MAPE %)")
    fig_bar = px.bar(
        df_komparasi, 
        x='Model', 
        y='MAPE', 
        color='Model',
        text_auto='.2f',
        color_discrete_map={'XGBoost_Teknikal': '#ff7f0e', 'XGBoost+Sentimen': '#2ca02c'}
    )
    fig_bar.update_layout(yaxis_title="MAPE (%)", height=300, showlegend=False, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=False)
