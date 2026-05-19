# 1. Pastikan import berada di PALING ATAS file
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
# import library lain yang Anda butuhkan...

# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    # Menggunakan nama file langsung jika diletakkan di folder utama GitHub
    df_raw = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")
    
    # Tambahkan proses preprocessing Anda di sini jika ada...
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal

# Memuat data ke dalam aplikasi
df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal = load_data()

# ==========================================
# 3. BAGIAN DISPLAY / METRIC (Periksa Baris 74)
# ==========================================
# Pastikan f-string ditutup dengan benar dan tidak menggantung ke baris baru
st.metric(label="Harga Penutupan Terakhir", value=f"Rp {harga_terakhir:,.2f}")
