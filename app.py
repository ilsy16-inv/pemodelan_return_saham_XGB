import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XGBoost · Volatilitas Saham LQ45 Energi",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body { font-family:'Inter',sans-serif !important; }
.block-container { padding-top:1rem !important; }

.hero {
    background:linear-gradient(135deg,#0d1117 0%,#161b27 40%,#1a1033 100%);
    border:1px solid rgba(99,102,241,0.25); border-radius:18px;
    padding:26px 30px; margin-bottom:20px;
    box-shadow:0 0 40px rgba(99,102,241,0.1);
}
.hero-title { color:#f1f5f9; font-size:1.65rem; font-weight:800; margin:0 0 4px; }
.hero-sub   { color:#64748b; font-size:0.85rem; margin:0; }
.badge {
    display:inline-block; background:rgba(99,102,241,0.15);
    border:1px solid rgba(99,102,241,0.3); color:#a5b4fc;
    border-radius:20px; padding:2px 11px; font-size:0.70rem;
    font-weight:600; margin:8px 3px 0 0; letter-spacing:.3px;
}
.badge-green {
    background:rgba(16,185,129,0.15); border-color:rgba(16,185,129,0.3); color:#6ee7b7;
}
.badge-orange {
    background:rgba(245,158,11,0.15); border-color:rgba(245,158,11,0.3); color:#fcd34d;
}

/* KPI Cards */
.kcard {
    background:linear-gradient(145deg,#161b27,#1c2132);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:18px 16px 14px; text-align:center;
    transition:.2s; height:100%;
}
.kcard:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.3); }
.kcard .ico { font-size:1.4rem; margin-bottom:5px; }
.kcard .val { font-size:1.5rem; font-weight:700; color:#f1f5f9; margin:3px 0; }
.kcard .lbl { font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:1.2px; }
.kcard .sub { font-size:0.75rem; color:#94a3b8; margin-top:3px; }
.kcard.c1 { border-top:3px solid #3b82f6; }
.kcard.c2 { border-top:3px solid #10b981; }
.kcard.c3 { border-top:3px solid #8b5cf6; }
.kcard.c4 { border-top:3px solid #f59e0b; }
.kcard.c5 { border-top:3px solid #ef4444; }
.kcard.c6 { border-top:3px solid #06b6d4; }

/* Section title */
.sec {
    font-size:.93rem; font-weight:600; color:#e2e8f0;
    margin:20px 0 10px; padding-left:10px;
    border-left:3px solid #6366f1;
}
.sec-sent { border-left-color:#f59e0b; }
.sec-eval { border-left-color:#10b981; }

/* Info boxes */
.ibox {
    background:rgba(59,130,246,0.07);
    border:1px solid rgba(59,130,246,0.18);
    border-radius:10px; padding:13px 15px;
    color:#93c5fd; font-size:.82rem; margin:8px 0;
    line-height:1.7;
}
.ibox-warn {
    background:rgba(245,158,11,0.07);
    border:1px solid rgba(245,158,11,0.2);
    color:#fcd34d;
}
.ibox-green {
    background:rgba(16,185,129,0.07);
    border:1px solid rgba(16,185,129,0.2);
    color:#6ee7b7;
}

/* Side boxes */
.sbox {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; padding:12px; margin-bottom:10px;
}

/* Live pill */
.live-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
    color:#34d399; border-radius:20px; padding:3px 12px;
    font-size:0.72rem; font-weight:600;
}
.live-dot {
    width:7px; height:7px; border-radius:50%;
    background:#10b981; animation:blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* Sentiment cards */
.sent-card {
    background:#1c2132; border-radius:10px;
    border:1px solid rgba(255,255,255,0.06);
    padding:14px; margin-bottom:8px;
}
.sent-pos { border-left:4px solid #10b981; }
.sent-neg { border-left:4px solid #ef4444; }
.sent-neu { border-left:4px solid #f59e0b; }

.footer {
    text-align:center; padding:16px 0 6px;
    color:#2d3748; font-size:.73rem;
    border-top:1px solid rgba(255,255,255,0.04); margin-top:22px;
}
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
[data-testid="stSidebar"] { background:#0f1117 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTA EMITEN
# ══════════════════════════════════════════════════════════════════════════════
EMITEN = {
    "ADRO": {"nama": "Adaro Energy",     "warna": "#6366f1"},
    "ADMR": {"nama": "Adaro Minerals",   "warna": "#10b981"},
    "ITMG": {"nama": "Indo Tambangraya", "warna": "#f59e0b"},
    "PTBA": {"nama": "Bukit Asam",       "warna": "#ef4444"},
    "MEDC": {"nama": "Medco Energi",     "warna": "#8b5cf6"},
    "AKRA": {"nama": "AKR Corporindo",   "warna": "#06b6d4"},
}
WARNA = {k: v["warna"] for k, v in EMITEN.items()}

def rgba(hex_c, a):
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def layout(h=340, title="", mt=36, mb=28):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=rgba("#1c2132", 0.9),
        font=dict(family="Inter", color="#94a3b8", size=11),
        xaxis=dict(gridcolor=rgba("#ffffff",0.04), linecolor=rgba("#ffffff",0.07)),
        yaxis=dict(gridcolor=rgba("#ffffff",0.04), linecolor=rgba("#ffffff",0.07)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=rgba("#ffffff",0.07), font=dict(size=11)),
        margin=dict(t=mt, b=mb, l=48, r=16),
        height=h,
        **({"title": dict(text=title, font=dict(size=13, color="#e2e8f0"))} if title else {}),
    )

# ══════════════════════════════════════════════════════════════════════════════
# NAMA KOLOM CSV (eksak sesuai file di repo)
# ══════════════════════════════════════════════════════════════════════════════
C_TGL     = "Tanggal"
C_EMITEN  = "Kode_Emiten"
C_AKTUAL  = "Volatilitas_Aktual"
C_PRED_S  = "Volatilitas_Pred_XGB_Sentimen_t1"
C_PRED_T  = "Volatilitas_Pred_XGB_Teknikal_t1"
C_ERR     = "Error_Abs"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_prediksi_sentimen():
    for p in ["hasil_prediksi_xgb_sentimen_t1.csv",
              "hasil_prediksi/hasil_prediksi_xgb_sentimen_t1.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df[C_TGL] = pd.to_datetime(df[C_TGL])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_prediksi_teknikal():
    for p in ["hasil_prediksi_xgb_teknikal_t1.csv",
              "hasil_prediksi/hasil_prediksi_xgb_teknikal_t1.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            df[C_TGL] = pd.to_datetime(df[C_TGL])
            return df
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_evaluasi_sentimen():
    for p in ["evaluasi_xgb_sentimen_t1.csv",
              "evaluasi/evaluasi_xgb_sentimen_t1.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_evaluasi_teknikal():
    for p in ["evaluasi_xgb_teknikal_t1.csv",
              "evaluasi/evaluasi_xgb_teknikal_t1.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_dataset():
    for p in ["dataset_merger_covid2020.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            if "Tanggal" in df.columns:
                df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            return df
    return pd.DataFrame()

df_sent  = load_prediksi_sentimen()
df_tech  = load_prediksi_teknikal()
df_eval_s = load_evaluasi_sentimen()
df_eval_t = load_evaluasi_teknikal()
df_raw   = load_dataset()

SENT_OK = len(df_sent) > 0
TECH_OK = len(df_tech) > 0

# ══════════════════════════════════════════════════════════════════════════════
# HELPER METRIK
# ══════════════════════════════════════════════════════════════════════════════
def get_m(df, keys, default=0.0):
    for k in keys:
        if k in df.columns:
            try: return float(df[k].iloc[0])
            except: pass
    return default

def calc_metrics(df_pred, col_pred):
    """Hitung RMSE, MAE, R² langsung dari dataframe prediksi."""
    if df_pred.empty or C_AKTUAL not in df_pred.columns or col_pred not in df_pred.columns:
        return 0.0, 0.0, 0.0
    a = df_pred[C_AKTUAL].values
    p = df_pred[col_pred].values
    rmse = float(np.sqrt(np.mean((a - p)**2)))
    mae  = float(np.mean(np.abs(a - p)))
    ss_r = np.sum((a - p)**2)
    ss_t = np.sum((a - a.mean())**2)
    r2   = float(1 - ss_r/ss_t) if ss_t != 0 else 0.0
    return rmse, mae, r2

RMSE_S, MAE_S, R2_S = (
    get_m(df_eval_s, ['RMSE','rmse']),
    get_m(df_eval_s, ['MAE','mae']),
    get_m(df_eval_s, ['R2','r2','R-Squared'])
) if not df_eval_s.empty else calc_metrics(df_sent, C_PRED_S)

RMSE_T, MAE_T, R2_T = (
    get_m(df_eval_t, ['RMSE','rmse']),
    get_m(df_eval_t, ['MAE','mae']),
    get_m(df_eval_t, ['R2','r2','R-Squared'])
) if not df_eval_t.empty else calc_metrics(df_tech, C_PRED_T)

# Kalau masih 0, hitung ulang dari data
if RMSE_S == 0 and SENT_OK: RMSE_S, MAE_S, R2_S = calc_metrics(df_sent, C_PRED_S)
if RMSE_T == 0 and TECH_OK: RMSE_T, MAE_T, R2_T = calc_metrics(df_tech, C_PRED_T)

MODEL_TERBAIK = "XGBoost + Sentimen" if RMSE_S <= RMSE_T else "XGBoost Teknikal"
RMSE_BEST     = min(RMSE_S, RMSE_T)

# Daftar emiten dari data
if SENT_OK and C_EMITEN in df_sent.columns:
    EMITEN_ADA = sorted(df_sent[C_EMITEN].unique().tolist())
else:
    EMITEN_ADA = list(EMITEN.keys())

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 16px">
      <div style="font-size:2rem">📈</div>
      <div style="font-size:.95rem;font-weight:700;color:#e2e8f0">Volatilitas LQ45 Energi</div>
      <div style="font-size:.7rem;color:#475569;margin-top:2px">XGBoost · Teknikal & Sentimen</div>
    </div>""", unsafe_allow_html=True)

    # Navigasi
    menu = st.radio("Menu", [
        "🏠 Dashboard",
        "📈 Prediksi Volatilitas",
        "📰 Analisis Sentimen",
        "📊 Evaluasi Model",
        "🗂 Data Mentah"
    ], label_visibility="collapsed")

    st.markdown("---")

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("**⚙️ Filter Emiten**")
    emiten_sel = st.multiselect(
        "Emiten", EMITEN_ADA, default=EMITEN_ADA,
        label_visibility="collapsed"
    )
    if not emiten_sel:
        emiten_sel = EMITEN_ADA
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.77rem;color:#94a3b8;line-height:2">
    🔹 <b style="color:#e2e8f0">Model:</b> XGBoost<br>
    🔹 <b style="color:#e2e8f0">Dataset:</b> Saham Energi LQ45<br>
    🔹 <b style="color:#e2e8f0">Periode:</b> Pandemi COVID-19<br>
    🔹 <b style="color:#e2e8f0">Target:</b> Volatilitas Return<br>
    🔹 <b style="color:#e2e8f0">Fitur 1:</b> 12 Indikator Teknikal<br>
    🔹 <b style="color:#e2e8f0">Fitur 2:</b> Skor Sentimen NLP
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sbox">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.77rem;color:#94a3b8;line-height:1.9">
    👩‍🎓 <b style="color:#e2e8f0">Peneliti</b><br>
    Manajemen Teknologi<br>
    SIMT — ITS Surabaya<br>
    <span style="color:#6366f1;font-weight:700">2026</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: filter dataframe per emiten
# ══════════════════════════════════════════════════════════════════════════════
def filt(df, emiten_list):
    if df.empty or C_EMITEN not in df.columns:
        return df
    return df[df[C_EMITEN].isin(emiten_list)].copy()

def pct_change_label(val, prev):
    if prev == 0: return "N/A", "neu"
    pct = (val - prev) / abs(prev) * 100
    return f"{'▲' if pct >= 0 else '▼'} {abs(pct):.2f}%", "up" if pct >= 0 else "down"

# ══════════════════════════════════════════════════════════════════════════════
# ── HALAMAN 1: DASHBOARD ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Dashboard":

    # Hero
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">📈 Dashboard Prediksi Volatilitas Return Saham</div>
      <div class="hero-sub">Pemodelan XGBoost — Saham Sektor Energi LQ45 — Periode COVID-19</div>
      <span class="badge">XGBoost</span>
      <span class="badge badge-green">Teknikal & Sentimen</span>
      <span class="badge badge-orange">LQ45 Energi</span>
      <span class="badge">{'✅ Data Loaded' if SENT_OK else '⚠️ Periksa CSV'}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row
    dfs_f = filt(df_sent, emiten_sel)
    dft_f = filt(df_tech, emiten_sel)

    vol_now  = dfs_f[C_AKTUAL].iloc[-1]  if SENT_OK and len(dfs_f) > 0 else 0
    vol_prev = dfs_f[C_AKTUAL].iloc[-2]  if SENT_OK and len(dfs_f) > 1 else vol_now
    vol_lbl, vol_cls = pct_change_label(vol_now, vol_prev)

    pred_s_last = dfs_f[C_PRED_S].iloc[-1] if SENT_OK and C_PRED_S in dfs_f.columns and len(dfs_f) > 0 else 0
    pred_t_last = dft_f[C_PRED_T].iloc[-1] if TECH_OK and C_PRED_T in dft_f.columns and len(dft_f) > 0 else 0

    mae_s_now = dfs_f[C_ERR].mean() if C_ERR in dfs_f.columns and len(dfs_f)>0 else MAE_S
    n_emiten  = len(emiten_sel)
    n_obs     = len(dfs_f) if SENT_OK else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_data = [
        (c1, "c1", "📉", "Volatilitas Aktual", f"{vol_now:.5f}", vol_lbl),
        (c2, "c2", "🤖", "Pred XGB Sentimen",  f"{pred_s_last:.5f}", f"RMSE {RMSE_S:.5f}"),
        (c3, "c3", "📊", "Pred XGB Teknikal",  f"{pred_t_last:.5f}", f"RMSE {RMSE_T:.5f}"),
        (c4, "c4", "🏆", "Model Terbaik",       MODEL_TERBAIK.split()[0]+" ✓", f"RMSE {RMSE_BEST:.5f}"),
        (c5, "c5", "🏢", "Emiten Dipilih",      str(n_emiten), "Sektor Energi"),
        (c6, "c6", "📋", "Total Observasi",     f"{n_obs:,}", "Data Uji"),
    ]
    for col, cls, ico, lbl, val, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kcard {cls}">
              <div class="ico">{ico}</div>
              <div class="lbl">{lbl}</div>
              <div class="val">{val}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Volatilitas aktual per emiten (multi-line)
    st.markdown('<div class="sec">📉 Volatilitas Aktual per Emiten</div>', unsafe_allow_html=True)

    if SENT_OK:
        fig_v = go.Figure()
        for em in emiten_sel:
            d = dfs_f[dfs_f[C_EMITEN] == em].sort_values(C_TGL) if C_EMITEN in dfs_f.columns else dfs_f
            if d.empty: continue
            warna = WARNA.get(em, "#6366f1")
            fig_v.add_trace(go.Scatter(
                x=d[C_TGL], y=d[C_AKTUAL], mode="lines",
                name=em, line=dict(color=warna, width=1.8),
                hovertemplate=f"<b>{em}</b><br>%{{x|%d %b %Y}}<br>Vol: %{{y:.5f}}<extra></extra>"
            ))
        fig_v.update_layout(**layout(h=300, mt=20, mb=30))
        fig_v.update_layout(hovermode="x unified",
                            xaxis_title="Tanggal", yaxis_title="Volatilitas Return")
        st.plotly_chart(fig_v, width='stretch')
    else:
        st.markdown('<div class="ibox ibox-warn">⚠️ File <code>hasil_prediksi_xgb_sentimen_t1.csv</code> tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py</div>', unsafe_allow_html=True)

    # ── Perbandingan Aktual vs Prediksi (kedua model, gabungan emiten)
    st.markdown('<div class="sec">🆚 Perbandingan Aktual vs Prediksi — Kedua Model</div>', unsafe_allow_html=True)

    if SENT_OK and TECH_OK:
        col_l, col_r = st.columns(2)
        for col_plot, df_p, col_pred, label, warna_p in [
            (col_l, dfs_f, C_PRED_S, "XGB + Sentimen", "#f59e0b"),
            (col_r, dft_f, C_PRED_T, "XGB Teknikal",   "#8b5cf6"),
        ]:
            with col_plot:
                fig_ap = go.Figure()
                for em in emiten_sel:
                    if C_EMITEN not in df_p.columns: break
                    d = df_p[df_p[C_EMITEN] == em].sort_values(C_TGL)
                    if d.empty: continue
                    fig_ap.add_trace(go.Scatter(
                        x=d[C_TGL], y=d[C_AKTUAL], mode="lines",
                        name=f"Aktual {em}", line=dict(color=WARNA.get(em,"#6366f1"), width=1.5),
                        showlegend=False,
                        hovertemplate=f"Aktual {em}: %{{y:.5f}}<extra></extra>"
                    ))
                    fig_ap.add_trace(go.Scatter(
                        x=d[C_TGL], y=d[col_pred], mode="lines",
                        name=f"Pred {em}", line=dict(color=warna_p, width=1.2, dash="dash"),
                        showlegend=False,
                        hovertemplate=f"Pred {em}: %{{y:.5f}}<extra></extra>"
                    ))
                fig_ap.update_layout(**layout(h=270, title=label, mt=30, mb=25))
                fig_ap.update_layout(hovermode="x unified")
                st.plotly_chart(fig_ap, width='stretch')

    # ── Insight box
    better_icon = "📰" if RMSE_S <= RMSE_T else "📊"
    st.markdown(f"""
    <div class="ibox ibox-green">
      🏆 <b>Model Terbaik: {MODEL_TERBAIK}</b> {better_icon}<br>
      RMSE Sentimen: <b>{RMSE_S:.5f}</b> &nbsp;|&nbsp;
      RMSE Teknikal: <b>{RMSE_T:.5f}</b> &nbsp;|&nbsp;
      R² Sentimen: <b>{R2_S:.4f}</b> &nbsp;|&nbsp;
      R² Teknikal: <b>{R2_T:.4f}</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── HALAMAN 2: PREDIKSI VOLATILITAS ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Prediksi Volatilitas":

    st.markdown("""
    <div class="hero" style="padding:20px 26px">
      <div class="hero-title" style="font-size:1.35rem">📈 Prediksi Volatilitas Return</div>
      <div class="hero-sub">Aktual vs Prediksi XGBoost — per Emiten</div>
    </div>""", unsafe_allow_html=True)

    # Filter
    cf1, cf2, cf3 = st.columns([2, 2, 3])
    with cf1:
        em_pilih = st.selectbox("Pilih Emiten", emiten_sel)
    with cf2:
        model_pilih = st.selectbox("Model", ["Keduanya", "XGB + Sentimen", "XGB Teknikal"])

    dfs_em = filt(df_sent, [em_pilih]).sort_values(C_TGL) if SENT_OK else pd.DataFrame()
    dft_em = filt(df_tech, [em_pilih]).sort_values(C_TGL) if TECH_OK else pd.DataFrame()

    if dfs_em.empty and dft_em.empty:
        st.warning(f"Tidak ada data untuk emiten {em_pilih}")
    else:
        # ── Chart utama
        fig_main = go.Figure()

        # Aktual (dari sentimen, sama saja aktualnya)
        src = dfs_em if not dfs_em.empty else dft_em
        fig_main.add_trace(go.Scatter(
            x=src[C_TGL], y=src[C_AKTUAL], mode="lines",
            name="Volatilitas Aktual",
            line=dict(color="#3b82f6", width=2.5),
            hovertemplate="%{x|%d %b %Y}<br>Aktual: %{y:.5f}<extra></extra>"
        ))

        if model_pilih in ["Keduanya", "XGB + Sentimen"] and not dfs_em.empty and C_PRED_S in dfs_em.columns:
            fig_main.add_trace(go.Scatter(
                x=dfs_em[C_TGL], y=dfs_em[C_PRED_S], mode="lines",
                name="Prediksi XGB Sentimen",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>Pred Sentimen: %{y:.5f}<extra></extra>"
            ))

        if model_pilih in ["Keduanya", "XGB Teknikal"] and not dft_em.empty and C_PRED_T in dft_em.columns:
            fig_main.add_trace(go.Scatter(
                x=dft_em[C_TGL], y=dft_em[C_PRED_T], mode="lines",
                name="Prediksi XGB Teknikal",
                line=dict(color="#8b5cf6", width=2, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>Pred Teknikal: %{y:.5f}<extra></extra>"
            ))

        fig_main.update_layout(**layout(h=380, mt=20, mb=40))
        fig_main.update_layout(
            hovermode="x unified",
            xaxis_title="Tanggal",
            yaxis_title="Volatilitas Return",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )
        st.plotly_chart(fig_main, width='stretch')

        # ── Scatter plot Aktual vs Prediksi
        st.markdown('<div class="sec">🎯 Scatter: Aktual vs Prediksi</div>', unsafe_allow_html=True)
        col_sc1, col_sc2 = st.columns(2)

        for col_sc, df_sc, col_p, title, warna_sc in [
            (col_sc1, dfs_em, C_PRED_S, "XGB Sentimen", "#f59e0b"),
            (col_sc2, dft_em, C_PRED_T, "XGB Teknikal",  "#8b5cf6"),
        ]:
            with col_sc:
                if df_sc.empty or col_p not in df_sc.columns:
                    st.info(f"Data {title} tidak tersedia")
                    continue
                a_arr = df_sc[C_AKTUAL].values
                p_arr = df_sc[col_p].values
                mn, mx = min(a_arr.min(), p_arr.min()), max(a_arr.max(), p_arr.max())
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=a_arr, y=p_arr, mode="markers",
                    marker=dict(color=warna_sc, size=5, opacity=0.65),
                    name=title,
                    hovertemplate="Aktual: %{x:.5f}<br>Pred: %{y:.5f}<extra></extra>"
                ))
                fig_sc.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
                    name="Ideal (y=x)", showlegend=False
                ))
                rmse_sc, mae_sc, r2_sc = calc_metrics(df_sc, col_p)
                fig_sc.update_layout(**layout(h=280, title=f"{title} — R²={r2_sc:.4f}", mt=36, mb=30))
                fig_sc.update_layout(xaxis_title="Aktual", yaxis_title="Prediksi")
                st.plotly_chart(fig_sc, width='stretch')

        # ── Error (Error_Abs) timeline
        st.markdown('<div class="sec">📉 Error Absolut per Tanggal</div>', unsafe_allow_html=True)
        if not dfs_em.empty and C_ERR in dfs_em.columns:
            fig_err = go.Figure()
            if model_pilih in ["Keduanya", "XGB + Sentimen"]:
                fig_err.add_trace(go.Bar(
                    x=dfs_em[C_TGL], y=dfs_em[C_ERR],
                    name="Error Sentimen",
                    marker_color=rgba("#f59e0b", 0.7)
                ))
            if model_pilih in ["Keduanya", "XGB Teknikal"] and not dft_em.empty and C_ERR in dft_em.columns:
                fig_err.add_trace(go.Bar(
                    x=dft_em[C_TGL], y=dft_em[C_ERR],
                    name="Error Teknikal",
                    marker_color=rgba("#8b5cf6", 0.7)
                ))
            fig_err.update_layout(**layout(h=240, mt=20, mb=30))
            fig_err.update_layout(barmode="group", hovermode="x unified",
                                  xaxis_title="Tanggal", yaxis_title="Error Absolut")
            st.plotly_chart(fig_err, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# ── HALAMAN 3: ANALISIS SENTIMEN ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📰 Analisis Sentimen":

    st.markdown("""
    <div class="hero" style="padding:20px 26px">
      <div class="hero-title" style="font-size:1.35rem">📰 Analisis Sentimen</div>
      <div class="hero-sub">Kontribusi fitur sentimen terhadap prediksi volatilitas XGBoost</div>
    </div>""", unsafe_allow_html=True)

    if not SENT_OK:
        st.markdown('<div class="ibox ibox-warn">⚠️ File prediksi sentimen tidak tersedia.</div>', unsafe_allow_html=True)
    else:
        dfs_f = filt(df_sent, emiten_sel)

        # ── Skor Sentimen vs Volatilitas Aktual (scatter)
        sent_cols = [c for c in df_raw.columns if any(k in c.lower() for k in ["sentimen","sentiment","sent","score"])] if not df_raw.empty else []

        if sent_cols:
            st.markdown('<div class="sec sec-sent">💬 Skor Sentimen vs Volatilitas Aktual</div>', unsafe_allow_html=True)
            sent_col_sel = st.selectbox("Kolom sentimen:", sent_cols)

            # Merge jika ada kolom tanggal/emiten yang sama
            if C_TGL in df_raw.columns and C_TGL in dfs_f.columns:
                df_merge = pd.merge(
                    dfs_f[[C_TGL, C_EMITEN, C_AKTUAL]],
                    df_raw[[C_TGL, sent_col_sel]],
                    on=C_TGL, how="inner"
                ).dropna()
            else:
                n_min = min(len(dfs_f), len(df_raw))
                df_merge = pd.DataFrame({
                    C_AKTUAL: dfs_f[C_AKTUAL].values[:n_min],
                    sent_col_sel: df_raw[sent_col_sel].values[:n_min]
                }).dropna()

            if not df_merge.empty:
                col_s1, col_s2 = st.columns([3, 2])
                with col_s1:
                    fig_ss = go.Figure()
                    fig_ss.add_trace(go.Scatter(
                        x=df_merge[sent_col_sel], y=df_merge[C_AKTUAL],
                        mode="markers",
                        marker=dict(
                            color=df_merge[C_AKTUAL],
                            colorscale=[[0,"#3b82f6"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                            size=5, opacity=0.7, showscale=True,
                            colorbar=dict(title="Volatilitas", thickness=12)
                        ),
                        hovertemplate="Sentimen: %{x:.3f}<br>Volatilitas: %{y:.5f}<extra></extra>"
                    ))
                    m, b = np.polyfit(df_merge[sent_col_sel], df_merge[C_AKTUAL], 1)
                    x_line = np.linspace(df_merge[sent_col_sel].min(), df_merge[sent_col_sel].max(), 50)
                    fig_ss.add_trace(go.Scatter(
                        x=x_line, y=m*x_line+b, mode="lines",
                        line=dict(color="#10b981", width=2, dash="dash"),
                        name=f"Trend (r={df_merge[[sent_col_sel,C_AKTUAL]].corr().iloc[0,1]:.3f})"
                    ))
                    fig_ss.update_layout(**layout(h=320, mt=20, mb=35))
                    fig_ss.update_layout(xaxis_title="Skor Sentimen", yaxis_title="Volatilitas Aktual")
                    st.plotly_chart(fig_ss, width='stretch')

                with col_s2:
                    corr_val = df_merge[[sent_col_sel, C_AKTUAL]].corr().iloc[0,1]
                    st.markdown(f"""
                    <div class="ibox" style="margin-top:10px">
                      <b>📊 Statistik Sentimen</b><br><br>
                      Rata-rata: <b>{df_merge[sent_col_sel].mean():.4f}</b><br>
                      Std Dev: <b>{df_merge[sent_col_sel].std():.4f}</b><br>
                      Min: <b>{df_merge[sent_col_sel].min():.4f}</b><br>
                      Max: <b>{df_merge[sent_col_sel].max():.4f}</b><br>
                      Korelasi dgn Volatilitas: <b>{corr_val:.4f}</b><br>
                      Arah: <b>{'Positif ↑' if corr_val > 0 else 'Negatif ↓'}</b>
                    </div>
                    <div class="ibox ibox-green" style="margin-top:10px">
                      💡 Korelasi <b>{abs(corr_val):.3f}</b> menunjukkan bahwa 
                      sentimen {'positif' if corr_val > 0 else 'negatif'} 
                      {'meningkatkan' if corr_val > 0 else 'menurunkan'} 
                      volatilitas pasar.
                    </div>
                    """, unsafe_allow_html=True)

        # ── Distribusi sentimen per emiten
        st.markdown('<div class="sec sec-sent">📊 Distribusi Error — Model Sentimen per Emiten</div>', unsafe_allow_html=True)

        if C_EMITEN in dfs_f.columns and C_ERR in dfs_f.columns:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                err_by_emiten = dfs_f.groupby(C_EMITEN)[C_ERR].agg(["mean","std","min","max"]).reset_index()
                err_by_emiten.columns = ["Emiten","MAE","Std","Min","Max"]
                fig_bar = go.Figure(go.Bar(
                    x=err_by_emiten["Emiten"],
                    y=err_by_emiten["MAE"],
                    marker_color=[WARNA.get(e,"#6366f1") for e in err_by_emiten["Emiten"]],
                    error_y=dict(type="data", array=err_by_emiten["Std"], visible=True),
                    hovertemplate="%{x}<br>MAE: %{y:.5f}<extra></extra>"
                ))
                fig_bar.update_layout(**layout(h=260, title="MAE per Emiten — XGB Sentimen", mt=36, mb=30))
                st.plotly_chart(fig_bar, width='stretch')

            with col_d2:
                fig_box = go.Figure()
                for em in emiten_sel:
                    d = dfs_f[dfs_f[C_EMITEN]==em][C_ERR].dropna()
                    if d.empty: continue
                    fig_box.add_trace(go.Box(
                        y=d, name=em,
                        marker_color=WARNA.get(em,"#6366f1"),
                        boxmean="sd"
                    ))
                fig_box.update_layout(**layout(h=260, title="Distribusi Error per Emiten", mt=36, mb=30))
                st.plotly_chart(fig_box, width='stretch')

        # ── Perbandingan Error: Sentimen vs Teknikal
        st.markdown('<div class="sec sec-sent">🆚 Error Sentimen vs Teknikal</div>', unsafe_allow_html=True)
        if TECH_OK and C_ERR in filt(df_tech, emiten_sel).columns:
            dft_f = filt(df_tech, emiten_sel)
            fig_cmp = go.Figure()
            for em in emiten_sel:
                ds = dfs_f[dfs_f[C_EMITEN]==em][C_ERR].values if C_EMITEN in dfs_f.columns else []
                dt = dft_f[dft_f[C_EMITEN]==em][C_ERR].values if C_EMITEN in dft_f.columns else []
                if len(ds) == 0 or len(dt) == 0: continue
                n = min(len(ds), len(dt))
                fig_cmp.add_trace(go.Scatter(
                    y=ds[:n], mode="lines", name=f"{em} Sentimen",
                    line=dict(color=WARNA.get(em,"#6366f1"), width=1.5)
                ))
                fig_cmp.add_trace(go.Scatter(
                    y=dt[:n], mode="lines", name=f"{em} Teknikal",
                    line=dict(color=WARNA.get(em,"#6366f1"), width=1.2, dash="dash")
                ))
            fig_cmp.update_layout(**layout(h=300, mt=20, mb=30))
            fig_cmp.update_layout(hovermode="x unified",
                                  yaxis_title="Error Absolut",
                                  xaxis_title="Index Observasi")
            st.plotly_chart(fig_cmp, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# ── HALAMAN 4: EVALUASI MODEL ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Evaluasi Model":

    st.markdown("""
    <div class="hero" style="padding:20px 26px">
      <div class="hero-title" style="font-size:1.35rem">📊 Evaluasi Performa Model</div>
      <div class="hero-sub">Perbandingan XGBoost Teknikal vs XGBoost Sentimen</div>
    </div>""", unsafe_allow_html=True)

    # ── Kartu metrik besar
    c1,c2,c3 = st.columns(3)
    for col, label, rmse, mae, r2, cls in [
        (c1, "XGB + Sentimen 📰", RMSE_S, MAE_S, R2_S, "c4"),
        (c2, "XGB Teknikal 📊",   RMSE_T, MAE_T, R2_T, "c3"),
        (c3, "Selisih RMSE 🔍",   abs(RMSE_S-RMSE_T), abs(MAE_S-MAE_T), abs(R2_S-R2_T), "c2"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kcard {cls}" style="text-align:left;padding:20px">
              <div style="font-size:.85rem;font-weight:700;color:#e2e8f0;margin-bottom:12px">{label}</div>
              <div style="font-size:.77rem;color:#94a3b8;line-height:2.2">
                RMSE: <b style="color:#f1f5f9;font-size:1rem">{rmse:.6f}</b><br>
                MAE: &nbsp;<b style="color:#f1f5f9;font-size:1rem">{mae:.6f}</b><br>
                R²: &nbsp;&nbsp;<b style="color:#f1f5f9;font-size:1rem">{r2:.5f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Tabel evaluasi per emiten
    st.markdown('<div class="sec sec-eval">📋 Evaluasi Per Emiten</div>', unsafe_allow_html=True)
    rows_eval = []
    for em in emiten_sel:
        ds = filt(df_sent, [em])
        dt = filt(df_tech, [em])
        if ds.empty and dt.empty: continue
        rs, ms, r2s = calc_metrics(ds, C_PRED_S)
        rt, mt_, r2t = calc_metrics(dt, C_PRED_T)
        rows_eval.append({
            "Emiten": em,
            "RMSE Sentimen": round(rs, 6),
            "MAE Sentimen":  round(ms, 6),
            "R² Sentimen":   round(r2s, 5),
            "RMSE Teknikal": round(rt, 6),
            "MAE Teknikal":  round(mt_, 6),
            "R² Teknikal":   round(r2t, 5),
            "Model Terbaik": "Sentimen" if rs <= rt else "Teknikal",
        })
    if rows_eval:
        df_eval_tbl = pd.DataFrame(rows_eval)
        st.dataframe(df_eval_tbl.style.background_gradient(
            subset=["RMSE Sentimen","RMSE Teknikal"], cmap="RdYlGn_r"
        ), width='stretch')

    # ── Bar chart RMSE per emiten
    if rows_eval:
        st.markdown('<div class="sec sec-eval">📊 RMSE per Emiten</div>', unsafe_allow_html=True)
        df_ev = pd.DataFrame(rows_eval)
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Bar(
            x=df_ev["Emiten"], y=df_ev["RMSE Sentimen"],
            name="RMSE Sentimen", marker_color="#f59e0b"
        ))
        fig_ev.add_trace(go.Bar(
            x=df_ev["Emiten"], y=df_ev["RMSE Teknikal"],
            name="RMSE Teknikal", marker_color="#8b5cf6"
        ))
        fig_ev.update_layout(**layout(h=300, mt=20, mb=30))
        fig_ev.update_layout(barmode="group", yaxis_title="RMSE", hovermode="x unified")
        st.plotly_chart(fig_ev, width='stretch')

        # ── Radar
        st.markdown('<div class="sec sec-eval">🕸 Radar Perbandingan Model (Normalisasi)</div>', unsafe_allow_html=True)
        cats = ["RMSE", "MAE", "1-R²"]
        vs = [RMSE_S, MAE_S, max(0, 1-R2_S)]
        vt = [RMSE_T, MAE_T, max(0, 1-R2_T)]
        mx = [max(a,b)+1e-9 for a,b in zip(vs,vt)]
        vs_n = [v/m for v,m in zip(vs,mx)]
        vt_n = [v/m for v,m in zip(vt,mx)]
        fig_r = go.Figure()
        for vals, name, color in [(vs_n,"XGB Sentimen","#f59e0b"),(vt_n,"XGB Teknikal","#8b5cf6")]:
            fig_r.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", name=name,
                fillcolor=rgba(color.lstrip("#") and color, 0.15),
                line=dict(color=color, width=2)
            ))
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor=rgba("#1c2132", 0.9),
                radialaxis=dict(gridcolor=rgba("#ffffff",0.08), color="#64748b"),
                angularaxis=dict(gridcolor=rgba("#ffffff",0.08), color="#94a3b8")
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
            margin=dict(t=30,b=30,l=40,r=40), height=340
        )
        col_rad, col_info = st.columns([3,2])
        with col_rad:
            st.plotly_chart(fig_r, width='stretch')
        with col_info:
            winner = "Sentimen" if RMSE_S <= RMSE_T else "Teknikal"
            st.markdown(f"""
            <div class="ibox ibox-green" style="margin-top:16px">
              🏆 <b>Kesimpulan Model</b><br><br>
              Model <b>XGB {winner}</b> menghasilkan RMSE lebih rendah
              ({RMSE_BEST:.6f}), artinya prediksi lebih mendekati nilai aktual volatilitas.<br><br>
              R² terbaik: <b>{max(R2_S, R2_T):.5f}</b><br>
              MAE terbaik: <b>{min(MAE_S, MAE_T):.6f}</b>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── HALAMAN 5: DATA MENTAH ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🗂 Data Mentah":

    st.markdown("""
    <div class="hero" style="padding:20px 26px">
      <div class="hero-title" style="font-size:1.35rem">🗂 Eksplorasi Data Mentah</div>
      <div class="hero-sub">Tabel prediksi dan evaluasi lengkap</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📰 Prediksi Sentimen",
        "📊 Prediksi Teknikal",
        "📋 Evaluasi",
        "🗃 Dataset Asli"
    ])

    with tab1:
        if SENT_OK:
            dfs_show = filt(df_sent, emiten_sel)
            st.caption(f"**{len(dfs_show):,} baris** · Prediksi XGBoost + Sentimen")
            st.dataframe(dfs_show, width='stretch')
            st.download_button("📥 Download CSV",
                data=dfs_show.to_csv(index=False),
                file_name="prediksi_xgb_sentimen.csv", mime="text/csv")
        else:
            st.warning("File prediksi sentimen tidak ditemukan.")

    with tab2:
        if TECH_OK:
            dft_show = filt(df_tech, emiten_sel)
            st.caption(f"**{len(dft_show):,} baris** · Prediksi XGBoost Teknikal")
            st.dataframe(dft_show, width='stretch')
            st.download_button("📥 Download CSV",
                data=dft_show.to_csv(index=False),
                file_name="prediksi_xgb_teknikal.csv", mime="text/csv")
        else:
            st.warning("File prediksi teknikal tidak ditemukan.")

    with tab3:
        c_es, c_et = st.columns(2)
        with c_es:
            st.caption("**Evaluasi — XGB Sentimen**")
            if not df_eval_s.empty:
                st.dataframe(df_eval_s, width='stretch')
            else:
                data_auto = {"Metrik":["RMSE","MAE","R²"],
                             "Nilai":[RMSE_S, MAE_S, R2_S]}
                st.dataframe(pd.DataFrame(data_auto), width='stretch')
        with c_et:
            st.caption("**Evaluasi — XGB Teknikal**")
            if not df_eval_t.empty:
                st.dataframe(df_eval_t, width='stretch')
            else:
                data_auto = {"Metrik":["RMSE","MAE","R²"],
                             "Nilai":[RMSE_T, MAE_T, R2_T]}
                st.dataframe(pd.DataFrame(data_auto), width='stretch')

    with tab4:
        if not df_raw.empty:
            st.caption(f"**{len(df_raw):,} baris** · Dataset historis COVID-19")
            st.dataframe(df_raw.tail(200), width='stretch')
            st.download_button("📥 Download Dataset",
                data=df_raw.to_csv(index=False),
                file_name="dataset_merger_covid2020.csv", mime="text/csv")
        else:
            st.warning("File dataset_merger_covid2020.csv tidak ditemukan.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  📈 XGBoost StockVision · Pemodelan Volatilitas Return Saham Energi LQ45 ·
  Teknikal &amp; Sentimen · © 2026 Academic Research
</div>
""", unsafe_allow_html=True)
