# ==============================================================================
# IMPORT LIBRARY
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Prediksi Saham",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    text-align: center;
}

.metric-title {
    font-size: 16px;
    color: #A0AEC0;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
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
# LOAD DATA
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
    st.error(f"Gagal membaca data: {e}")
    st.stop()

# ==============================================================================
# PREPARASI DATA
# ==============================================================================
if 'Close' in df_raw.columns:
    harga_terakhir = df_raw['Close'].iloc[-1]
else:
    harga_terakhir = 0

rmse_sentimen = df_eval_sentimen['RMSE'].iloc[0]
rmse_teknikal = df_eval_teknikal['RMSE'].iloc[0]

# Tentukan model terbaik
if rmse_sentimen < rmse_teknikal:
    model_terbaik = "XGBoost + Sentimen"
else:
    model_terbaik = "XGBoost Teknikal"

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("📊 Navigation")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard Utama",
        "📈 Visualisasi Prediksi",
        "📋 Data & Evaluasi"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
Dashboard Prediksi Volatilitas Return Saham
menggunakan metode XGBoost berbasis:
- Sentimen
- Teknikal
""")

# ==============================================================================
# HEADER
# ==============================================================================
st.title("📈 Dashboard Prediksi Volatilitas Return Saham")
st.markdown("""
### Analisis Prediksi Saham Sektor Energi LQ45  
Menggunakan Metode **XGBoost** dan **Analisis Sentimen**
""")

st.markdown("---")

# ==============================================================================
# DASHBOARD UTAMA
# ==============================================================================
if menu == "🏠 Dashboard Utama":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Harga Penutupan Terakhir</div>
            <div class="metric-value">Rp {harga_terakhir:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE Model Sentimen</div>
            <div class="metric-value">{rmse_sentimen:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE Model Teknikal</div>
            <div class="metric-value">{rmse_teknikal:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🏆 Insight Model")

    st.success(f"""
    Model terbaik berdasarkan nilai RMSE adalah:
    **{model_terbaik}**
    """)

    st.markdown("---")

    st.subheader("📌 Ringkasan Penelitian")

    st.write("""
    Dashboard ini digunakan untuk membandingkan performa
    model prediksi volatilitas return saham menggunakan:

    - XGBoost berbasis analisis sentimen
    - XGBoost berbasis indikator teknikal

    Dataset berasal dari saham sektor energi indeks LQ45
    selama periode pandemi COVID-19.
    """)

# ==============================================================================
# VISUALISASI
# ==============================================================================
elif menu == "📈 Visualisasi Prediksi":

    st.subheader("📈 Grafik Perbandingan Prediksi")

    df_plot = pd.DataFrame()

    if 'Close' in df_raw.columns:
        df_plot['Aktual'] = df_raw['Close'].tail(len(df_pred_sentimen)).values

    if 'Prediction' in df_pred_sentimen.columns:
        df_plot['Prediksi Sentimen'] = df_pred_sentimen['Prediction']

    if 'Prediction' in df_pred_teknikal.columns:
        df_plot['Prediksi Teknikal'] = df_pred_teknikal['Prediction']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=df_plot['Aktual'],
        mode='lines',
        name='Data Aktual'
    ))

    fig.add_trace(go.Scatter(
        y=df_plot['Prediksi Sentimen'],
        mode='lines',
        name='XGBoost + Sentimen'
    ))

    fig.add_trace(go.Scatter(
        y=df_plot['Prediksi Teknikal'],
        mode='lines',
        name='XGBoost Teknikal'
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        title="Perbandingan Prediksi Model",
        xaxis_title="Periode",
        yaxis_title="Harga Saham"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Grafik menunjukkan perbandingan antara:
    - Harga aktual saham
    - Prediksi model berbasis sentimen
    - Prediksi model berbasis teknikal
    """)

# ==============================================================================
# DATA & EVALUASI
# ==============================================================================
elif menu == "📋 Data & Evaluasi":

    st.subheader("📋 Tabel Data dan Evaluasi")

    tab1, tab2, tab3 = st.tabs([
        "📄 Dataset",
        "🤖 Model Sentimen",
        "💻 Model Teknikal"
    ])

    # ==========================================================================
    # TAB DATASET
    # ==========================================================================
    with tab1:

        st.write("### Dataset Utama")

        st.dataframe(df_raw.tail(20), use_container_width=True)

        st.download_button(
            label="⬇ Download Dataset",
            data=df_raw.to_csv(index=False),
            file_name="dataset.csv",
            mime="text/csv"
        )

    # ==========================================================================
    # TAB SENTIMEN
    # ==========================================================================
    with tab2:

        st.write("### Hasil Prediksi Sentimen")

        col1, col2 = st.columns([2,1])

        with col1:
            st.dataframe(df_pred_sentimen.head(20),
                         use_container_width=True)

        with col2:
            st.dataframe(df_eval_sentimen,
                         use_container_width=True)

    # ==========================================================================
    # TAB TEKNIKAL
    # ==========================================================================
    with tab3:

        st.write("### Hasil Prediksi Teknikal")

        col1, col2 = st.columns([2,1])

        with col1:
            st.dataframe(df_pred_teknikal.head(20),
                         use_container_width=True)

        with col2:
            st.dataframe(df_eval_teknikal,
                         use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")

st.caption("""
Developed for Research Purposes | 
Prediksi Volatilitas Return Saham menggunakan XGBoost
""")
