# ==========================================
# 2. FUNGSI UNTUK MEMUAT DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    # Mengubah jalur agar langsung membaca file di folder utama
    df_raw = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")
    
    # Memastikan format tanggal sesuai
    df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
    df_pred_sentimen['Tanggal'] = pd.to_datetime(df_pred_sentimen['Tanggal'])
    df_pred_teknikal['Tanggal'] = pd.to_datetime(df_pred_teknikal['Tanggal'])
    
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal
