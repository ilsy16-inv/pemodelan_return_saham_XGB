# ==============================================================================
# 1. IMPORT LIBRARY (Wajib Berada di Paling Atas File)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Konfigurasi halaman utama Streamlit
st.set_page_config(
    page_title="Pemodelan Volatilitas Return Saham",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. FUNGSI UNTUK MEMUAT DATA
# ==============================================================================
@st.cache_data(ttl=3600)
def load_data():
    """
    Fungsi untuk membaca data CSV langsung dari folder utama (root) GitHub.
    """
    df_raw = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")
    
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal

# Eksekusi fungsi muat data dengan pengaman (Try-Except)
try:
    df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal = load_data()
except FileNotFoundError as e:
    st.error(f"❌ **Gagal memuat file data:** {e}")
    st.info("Pastikan semua file `.csv` berikut sudah di-upload ke folder utama repositori GitHub Anda:\n"
            "- `dataset_merger_covid2020.csv`\n"
            "- `hasil_prediksi_xgb_sentimen_t1.csv`\n"
            "- `hasil_prediksi_xgb_teknikal_t1.csv`\n"
            "- `evaluasi_xgb_sentimen_t1.csv`\n"
            "- `evaluasi_xgb_teknikal_t1.csv`")
    st.stop()

# ==============================================================================
# 3. INTERFACE APLIKASI & RINGKASAN METRIK
# ==============================================================================
st.title("📈 Pemodelan Prediksi Volatilitas Return Saham Sektor Energi (LQ45)")
st.subheader("Berdasarkan Analisis Sentimen Menggunakan Metode XGBoost dan LSTM")
st.markdown("---")

# --- AMBIL NILAI UNTUK METRIK DENGAN AMAN ---
# 1. Mengambil harga penutupan terakhir dari df_raw
try:
    # Mengambil nilai baris terakhir dari kolom 'Close'. 
    # (Sesuaikan huruf besar/kecil 'Close' dengan kolom asli di file CSV Anda)
    harga_terakhir = df_raw['Close'].iloc[-1]
except Exception:
    harga_terakhir = 0.0

# 2. Mengambil nilai RMSE dari file evaluasi model sentimen
try:
    rmse_sentimen = df_eval_sentimen['RMSE'].iloc[0]
except Exception:
    rmse_sentimen = 0.0

# 3. Mengambil nilai RMSE dari file evaluasi model teknikal
try:
    rmse_teknikal = df_eval_teknikal['RMSE'].iloc[0]
except Exception:
    rmse_teknikal = 0.0


# --- MENAMPILKAN METRIK PADA DASHBOARD ---
col1, col2, col3 = st.columns(3)

with col1:
    # Memperbaiki f-string agar tertutup dengan benar dalam satu baris
    st.metric(label="Harga Penutupan Terakhir", value=f"Rp {harga_terakhir:,.2f}")

with col2:
    st.metric(label="RMSE Model Sentimen (XGBoost)", value=f"{rmse_sentimen:.4f}")

with col3:
    st.metric(label="RMSE Model Teknikal (XGBoost)", value=f"{rmse_teknikal:.4f}")

st.markdown("---")

# ==============================================================================
# 4. KODE VISUALISASI / GRAFIK ANDA (Lanjutkan di bawah ini)
# ==============================================================================
# Tempatkan kode chart/plotly/tabel Anda di area ini...
