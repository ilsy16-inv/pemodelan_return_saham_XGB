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
    Pastikan nama file di bawah ini sesuai dengan yang ada di repositori Anda.
    """
    df_raw = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")
    
    return df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal

# Eksekusi fungsi muat data dengan proteksi (Try-Except)
try:
    df_raw, df_pred_sentimen, df_pred_teknikal, df_eval_sentimen, df_eval_teknikal = load_data()
except FileNotFoundError as e:
    st.error(f"❌ **Gagal memuat file data:** {e}")
    st.info("Pastikan semua file `.csv` berikut sudah di-upload ke folder utama (root) di GitHub Anda:\n"
            "- `dataset_merger_covid2020.csv`\n"
            "- `hasil_prediksi_xgb_sentimen_t1.csv`\n"
            "- `hasil_prediksi_xgb_teknikal_t1.csv`\n"
            "- `evaluasi_xgb_sentimen_t1.csv`\n"
            "- `evaluasi_xgb_teknikal_t1.csv`")
    st.stop()

# ==============================================================================
# 3. INTERFACE APLIKASI & PREPARASI VARIABEL METRIK
# ==============================================================================
st.title("📈 Pemodelan Prediksi Volatilitas Return Saham Sektor Energi (LQ45)")
st.subheader("Berdasarkan Analisis Sentimen Menggunakan Metode XGBoost dan LSTM")
st.markdown("---")

# --- AMBIL NILAI UNTUK METRIK DENGAN AMAN (Mencegah NameError) ---

# 1. Mengambil harga penutupan terakhir dari df_raw
try:
    # Mengambil nilai baris terakhir dari kolom 'Close'.
    # (Catatan: Jika nama kolom di file CSV Anda menggunakan huruf kecil semua seperti 'close' atau nama lain, sesuaikan tulisan di dalam tanda petik)
    if 'Close' in df_raw.columns:
        harga_terakhir = df_raw['Close'].iloc[-1]
    elif 'close' in df_raw.columns:
        harga_terakhir = df_raw['close'].iloc[-1]
    else:
        # Jika kolom tidak ditemukan, ambil kolom numerik terakhir atau gunakan nilai fallback
        harga_terakhir = df_raw.select_dtypes(include=[np.number]).iloc[-1, 0]
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
    # Memperbaiki penulisan f-string agar tertutup dengan benar dalam satu baris kode tunggal
    st.metric(label="Harga Penutupan Terakhir", value=f"Rp {harga_terakhir:,.2f}")

with col2:
    st.metric(label="RMSE Model Sentimen (XGBoost)", value=f"{rmse_sentimen:.4f}")

with col3:
    st.metric(label="RMSE Model Teknikal (XGBoost)", value=f"{rmse_teknikal:.4f}")

st.markdown("---")
# ==============================================================================
# 4. BAGIAN VISUALISASI DAN KODE GRAFIK ANDA (PILIHAN INTERAKTIF)
# ==============================================================================

# Membuat navigasi pilihan di sidebar atau halaman utama
pilihan_tampilan = st.radio(
    "Pilih Fitur Dashboard:",
    ["📈 Grafik Perbandingan Prediksi vs Aktual", "📊 Tabel Data & Evaluasi Model"]
)

if pilihan_tampilan == "📈 Grafik Perbandingan Prediksi vs Aktual":
    st.subheader("Tren Harga Aktual vs Hasil Prediksi Model")
    
    # Menyiapkan DataFrame gabungan untuk visualisasi grafik
    try:
        df_grafik = pd.DataFrame()
        
        # 1. Masukkan data aktual (ambil kolom Tanggal/Date dan Close jika ada)
        # Asumsi: Ada kolom 'Date' atau 'Tanggal' di dataset Anda. Jika tidak ada, menggunakan index.
        if 'Date' in df_raw.columns:
            df_grafik['Tanggal'] = pd.to_datetime(df_raw['Date'])
            df_grafik.set_index('Tanggal', inplace=True)
        elif 'Tanggal' in df_raw.columns:
            df_grafik['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
            df_grafik.set_index('Tanggal', inplace=True)
            
        # Isi kolom harga asli
        if 'Close' in df_raw.columns:
            df_grafik['Data Aktual'] = df_raw['Close']
        elif 'close' in df_raw.columns:
            df_grafik['Data Aktual'] = df_raw['close']
            
        # 2. Masukkan hasil prediksi XGBoost dari file t1 Anda
        # (Sesuaikan nama kolom target prediksi di file CSV Anda, contoh di bawah menggunakan kolom 'Prediction' atau 'Prediksi')
        if 'Prediction' in df_pred_sentimen.columns:
            df_grafik['XGBoost + Sentimen'] = df_pred_sentimen['Prediction'].values
        elif 'Prediksi' in df_pred_sentimen.columns:
            df_grafik['XGBoost + Sentimen'] = df_pred_sentimen['Prediksi'].values
            
        if 'Prediction' in df_pred_teknikal.columns:
            df_grafik['XGBoost Teknikal Murni'] = df_pred_teknikal['Prediction'].values
        elif 'Prediksi' in df_pred_teknikal.columns:
            df_grafik['XGBoost Teknikal Murni'] = df_pred_teknikal['Prediksi'].values

        # Menampilkan Line Chart Interaktif bawaan Streamlit
        if not df_grafik.empty:
            st.line_chart(df_grafik)
            st.caption("💡 *Gunakan scroll mouse atau drag pada grafik untuk melakukan zoom-in/zoom-out.*")
        else:
            st.warning("⚠️ Kolom data aktual atau prediksi tidak cocok. Menampilkan chart fallback otomatis:")
            # Jika kolom tidak pas, tampilkan visualisasi seadanya dari file prediksi sentimen
            st.line_chart(df_pred_sentimen.select_dtypes(include=[np.number]).iloc[:, :2])
            
    except Exception as e:
        st.error(f"Gagal memetakan grafik: {e}")
        st.info("Pastikan panjang baris data aktual dan data hasil prediksi Anda sama (sama-sama berukuran data testing).")

elif pilihan_tampilan == "📊 Tabel Data & Evaluasi Model":
    st.subheader("Detail Data & Metrik Performa")
    
    # Membuat tab terpisah untuk merapikan tampilan tabel
    tab1, tab2, tab3 = st.tabs(["📄 Data Aktual (Raw)", "🤖 Evaluasi Model Sentimen", "💻 Evaluasi Model Teknikal"])
    
    with tab1:
        st.write("**Sampel 10 Data Terakhir (Dataset Merger):**")
        st.dataframe(df_raw.tail(10), use_container_width=True)
        
    with tab2:
        st.write("**Hasil Prediksi & Metrik Evaluasi (XGBoost + Analisis Sentimen):**")
        col_t2_1, col_t2_2 = st.columns([2, 1])
        with col_t2_1:
            st.write("Tabel Prediksi:")
            st.dataframe(df_pred_sentimen.head(10), use_container_width=True)
        with col_t2_2:
            st.write("Nilai Error Lengkap:")
            st.dataframe(df_eval_sentimen, use_container_width=True)
            
    with tab3:
        st.write("**Hasil Prediksi & Metrik Evaluasi (XGBoost Teknikal Saham):**")
        col_t3_1, col_t3_2 = st.columns([2, 1])
        with col_t3_1:
            st.write("Tabel Prediksi:")
            st.dataframe(df_pred_teknikal.head(10), use_container_width=True)
        with col_t3_2:
            st.write("Nilai Error Lengkap:")
            st.dataframe(df_eval_teknikal, use_container_width=True)
