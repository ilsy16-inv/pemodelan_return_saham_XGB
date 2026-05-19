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
# 2. FUNGSI UNTUK MEMUAT DATA (DENGAN CACHING)
# ==========================================
@st.cache_data(ttl=3600)  # Data disimpan di cache selama 1 jam agar loading cepat
def load_data():
    # Menyesuaikan dengan susunan folder baru Anda
    df_raw = pd.read_csv("data/raw/dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("data/processed/hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("data/processed/hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("data/processed/evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("data/processed/evaluasi_xgb_teknikal_t1.csv")
    
    # Pastikan format tanggal sesuai tipe datetime
    df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
    df_pred_sentimen['Tanggal'] = pd.to_datetime(df_pred_sentimen['Tanggal'])
    df_pred_teknikal['Tanggal'] = pd.to_datetime(df_pred_teknikal['Tanggal'])
    
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal

# Load data ke aplikasi
df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal = load_data()

# ==========================================
# 3. SIDEBAR CONTROLLER (FILTER INTERAKTIF)
# ==========================================
st.sidebar.header("📊 Filter Analisis")

# Filter Pilih Emiten (ADRO, PTBA, ITMG, AKRA, MEDC, ADMR)
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

# Filter Data Berdasarkan Pilihan User
start_date, end_date = pd.to_datetime(pilihan_tanggal[0]), pd.to_datetime(pilihan_tanggal[1])

sub_raw = df_raw[(df_raw['Kode_Emiten'] == pilihan_emiten) & (df_raw['Tanggal'].between(start_date, end_date))]
sub_pred_s = df_pred_sentimen[(df_pred_sentimen['Kode_Emiten'] == pilihan_emiten) & (df_pred_sentimen['Tanggal'].between(start_date, end_date))]
sub_pred_t = df_pred_teknikal[(df_pred_teknikal['Kode_Emiten'] == pilihan_emiten) & (df_pred_teknikal['Tanggal'].between(start_date, end_date))]

# ==========================================
# 4. KONTEN UTAMA DASHBOARD
# ==========================================
st.title("📈 Dashboard Prediksi Volatilitas Return Saham Sektor Energi (LQ45)")
st.markdown(f"### Analisis Komparatif XGBoost Berdasarkan Indikator Teknikal & Sentimen Berita")
st.markdown("---")

# --- KARTU KPI UTAMA (METRICS CARD) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    harga_terakhir = sub_raw['Close'].iloc[-1] if not sub_raw.empty else 0
    st.metric(label="Harga Penutupan Terakhir", value=f"Rp
