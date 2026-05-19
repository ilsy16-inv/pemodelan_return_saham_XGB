# ==============================================================================
# IMPORT LIBRARY
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# KONFIGURASI HALAMAN (Wajib berada di perintah pertama Streamlit)
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Prediksi Saham Sektor Energi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS (Tampilan Gelap Modern & Kartu Kustom)
# ==============================================================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1c1f26;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    text-align: center;
    border: 1px solid #2d3139;
    margin-bottom: 15px;
}
.metric-title {
    font-size: 15px;
    color: #A0AEC0;
    font-weight: 500;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #ffffff;
}
.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD DATA (Menggunakan Fungsi Cache)
# ==============================================================================
@st.cache_data
def load_data():
    df_raw = pd.read_csv("dataset_merger_covid2020.csv")
    df_pred_sentimen = pd.read_csv("hasil_prediksi_xgb_sentimen_t1.csv")
    df_pred_teknikal = pd.read_csv("hasil_prediksi_xgb_teknikal_t1.csv")
    df_eval_sentimen = pd.read_csv("evaluasi_xgb_sentimen_t1.csv")
    df_eval_teknikal = pd.read_csv("evaluasi_xgb_teknikal_t1.csv")

    return (
        df_raw,
        df_pred_sentimen,
        df_pred_teknikal,
        df_eval_sentimen,
        df_eval_teknikal
    )

try:
    (
        df_raw,
        df_pred_sentimen,
        df_pred_teknikal,
        df_eval_sentimen,
        df_eval_teknikal
    ) = load_data()
except Exception as e:
    st.error(f"⚠️ Gagal membaca berkas data CSV: {e}")
    st.info("Pastikan semua file CSV berada di folder yang sama dengan file app.py Anda di GitHub.")
    st.stop()

# ==============================================================================
# PREPARASI DATA & PERHITUNGAN
# ==============================================================================
if 'Close' in df_raw.columns:
    harga_terakhir = df_raw['Close'].iloc[-1]
else:
    harga_terakhir = 0

rmse_sentimen = df_eval_sentimen['RMSE'].iloc[0] if 'RMSE' in df_eval_sentimen.columns else 0
rmse_teknikal = df_eval_teknikal['RMSE'].iloc[0] if 'RMSE' in df_eval_teknikal.columns else 0

# Menentukan model terbaik otomatis
if rmse_sentimen < rmse_teknikal:
    model_terbaik = "XGBoost + Analisis Sentimen 📰"
    warna_insight = "success"
else:
    model_terbaik = "XGBoost Indikator Teknikal 💻"
    warna_insight = "info"

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("📊 Navigasi Utama")

menu = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    [
        "🏠 Dashboard Utama",
        "📈 Visualisasi Prediksi",
        "📋 Data & Evaluasi"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Deskripsi Proyek")
st.sidebar.info("""
**Dashboard Prediksi Volatilitas Return Saham** Aplikasi perbandingan performa algoritma **XGBoost** berbasis data internal (Teknikal Pasar) dan data eksternal (Sentimen Berita).
""")

# ==============================================================================
# HEADER UTAMA DASHBOARD
# ==============================================================================
st.title("📈 Dashboard Analisis & Prediksi Volatilitas Return Saham")
st.markdown("### Studi Kasus: Saham Sektor Energi Indeks LQ45 (Periode Pandemi COVID-19)")
st.markdown("---")

# ==============================================================================
# HALAMAN 1: DASHBOARD UTAMA
# ==============================================================================
if menu == "🏠 Dashboard Utama":
    
    # Baris KPI Cards (Menggunakan layout kolom)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Harga Penutupan Terakhir</div>
            <div class="metric-value">Rp {harga_terakhir:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📉 RMSE Model Sentimen</div>
            <div class="metric-value">{rmse_sentimen:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💻 RMSE Model Teknikal</div>
            <div class="metric-value">{rmse_teknikal:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🏆 Kesimpulan Akurasi Model Terbaik")
    if warna_insight == "success":
        st.success(f"Berdasarkan tingkat galat terendah (Nilai RMSE terkecil), model terbaik adalah: **{model_terbaik}**")
    else:
        st.info(f"Berdasarkan tingkat galat terendah (Nilai RMSE terkecil), model terbaik adalah: **{model_terbaik}**")

    st.markdown("---")
    st.subheader("📌 Ringkasan Ringkas Penelitian")
    
    col_text, col_img = st.columns([3, 1])
    with col_text:
        st.write("""
        Sistem dashboard interaktif ini memetakan sekaligus menguji keandalan performa model predictive analytics 
        **XGBoost (Extreme Gradient Boosting)** untuk memproyeksikan volatilitas imbal hasil instrumen saham. 
        
        Penelitian difokuskan pada emiten saham kategori sektor energi yang terdaftar dalam indeks LQ45 selama masa turbulensi pasar akibat **Pandemi COVID-19**. Komparasi dilakukan secara langsung antara:
        1. **Model Pendekatan Sentimen**: Mengintegrasikan ekstraksi opini atau sentimen publik/berita eksternal.
        2. **Model Pendekatan Teknikal**: Mengandalkan pola data historis deret waktu kuantitatif dari pergerakan pasar murni.
        """)

# ==============================================================================
# HALAMAN 2: VISUALISASI GRAFIK
# ==============================================================================
elif menu == "📈 Visualisasi Prediksi":

    st.subheader("📊 Grafik Interaktif Perbandingan Nilai Aktual vs Hasil Prediksi")

    # Ambil data pengujian terakhir
    df_plot = pd.DataFrame()

    if 'Close' in df_raw.columns:
        df_plot['Aktual'] = df_raw['Close'].tail(len(df_pred_sentimen)).values

    if 'Prediction' in df_pred_sentimen.columns:
        df_plot['Prediksi Sentimen'] = df_pred_sentimen['Prediction'].values

    if 'Prediction' in df_pred_teknikal.columns:
        df_plot['Prediksi Teknikal'] = df_pred_teknikal['Prediction'].values

    # Inisialisasi Objek Grafik Plotly
    fig = go.Figure()

    # PERBAIKAN UTAMA: Validasi keberadaan kolom sebelum digambar agar tidak KeyError
    if 'Aktual' in df_plot.columns:
        fig.add_trace(go.Scatter(
            y=df_plot['Aktual'],
            mode='lines',
            name='Data Aktual (Pasar)',
            line=dict(color='#00CC96', width=2.5)
        ))

    if 'Prediksi Sentimen' in df_plot.columns:
        fig.add_trace(go.Scatter(
            y=df_plot['Prediksi Sentimen'],
            mode='lines',
            name='Prediksi Model XGBoost + Sentimen',
            line=dict(color='#636EFA', width=2, dash='dash')
        ))

    if 'Prediksi Teknikal' in df_plot.columns:
        fig.add_trace(go.Scatter(
            y=df_plot['Prediksi Teknikal'],
            mode='lines',
            name='Prediksi Model XGBoost Teknikal Only',
            line=dict(color='#EF553B', width=2, dash='dot')
        ))

    # Kustomisasi Layout Grafik ke Tema Gelap Terminal Finansial
    fig.update_layout(
        template="plotly_dark",
        height=550,
        title=dict(text="Simulasi Hasil Prediksi Model Terhadap Data Aktual", font=dict(size=18)),
        xaxis_title="Index Baris Data Pengujian (Timeline)",
        yaxis_title="Nilai / Harga Saham",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(ylink='top', y=0.99, x=0.01, xanchor='left')
    )

    # Tampilkan Grafik ke Web Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **Petunjuk Interaksi Grafik:**
    Anda dapat mendekatkan kursor untuk melihat rincian angka (*Hover*), melakukan *zoom-in* dengan menahan klik kiri mouse, 
    atau menyembunyikan garis grafik tertentu dengan cara mengklik nama legenda grafik di bagian atas.
    """)

# ==============================================================================
# HALAMAN 3: STRUKTUR TABEL DATA & EVALUASI
# ==============================================================================
elif menu == "📋 Data & Evaluasi":

    st.subheader("📋 Manajemen Eksplorasi Data & Matriks Evaluasi Model")

    # Menggunakan komponen TABS agar UI ringkas dan mewah
    tab1, tab2, tab3 = st.tabs([
        "📁 Dataset Mentah (Historical)",
        "🤖 Hasil & Evaluasi Model Sentimen",
        "💻 Hasil & Evaluasi Model Teknikal"
    ])

    # --- TAB 1: DATASET ---
    with tab1:
        st.markdown("### 📄 Dataset Terintegrasi Historis")
        st.dataframe(df_raw.tail(30), use_container_width=True)
        
        st.download_button(
            label="📥 Download Dataset Utama (CSV)",
            data=df_raw.to_csv(index=False),
            file_name="dataset_clean_covid.csv",
            mime="text/csv"
        )

    # --- TAB 2: MODEL SENTIMEN ---
    with tab2:
        st.markdown("### 📰 Model Kombinasi Analisis Sentimen Eksternal")
        
        col_table, col_eval = st.columns([2, 1])
        with col_table:
            st.markdown("**Sampel Hasil Prediksi (20 Baris Pertama):**")
            st.dataframe(df_pred_sentimen.head(20), use_container_width=True)
        with col_eval:
            st.markdown("**Nilai Matriks Evaluasi (Eror):**")
            st.dataframe(df_eval_sentimen, use_container_width=True)

    # --- TAB 3: MODEL TEKNIKAL ---
    with tab3:
        st.markdown("### 💻 Model Konvensional Indikator Teknikal Pasar")
        
        col_table2, col_eval2 = st.columns([2, 1])
        with col_table2:
            st.markdown("**Sampel Hasil Prediksi (20 Baris Pertama):**")
            st.dataframe(df_pred_teknikal.head(20), use_container_width=True)
        with col_eval2:
            st.markdown("**Nilai Matriks Evaluasi (Eror):**")
            st.dataframe(df_eval_teknikal, use_container_width=True)

# ==============================================================================
# FOOTER (Tampilan Bawah)
# ==============================================================================
st.markdown("---")
st.caption("© 2026 Developed for Academic Research Purposes | Pemodelan Machine Learning Return Saham XGBoost")
