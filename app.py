import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, warnings, io
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XGBoost · Volatilitas Saham LQ45 Energi",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*,html,body{font-family:'Inter',sans-serif!important}
.block-container{padding-top:1rem!important}
.hero{background:linear-gradient(135deg,#0d1117 0%,#161b27 40%,#1a1033 100%);
  border:1px solid rgba(99,102,241,.25);border-radius:18px;
  padding:26px 30px;margin-bottom:18px;box-shadow:0 0 40px rgba(99,102,241,.1)}
.hero-title{color:#f1f5f9;font-size:1.6rem;font-weight:800;margin:0 0 4px}
.hero-sub{color:#64748b;font-size:.85rem;margin:0}
.badge{display:inline-block;background:rgba(99,102,241,.15);
  border:1px solid rgba(99,102,241,.3);color:#a5b4fc;border-radius:20px;
  padding:2px 11px;font-size:.69rem;font-weight:600;margin:8px 3px 0 0;letter-spacing:.3px}
.badge-g{background:rgba(16,185,129,.15);border-color:rgba(16,185,129,.3);color:#6ee7b7}
.badge-y{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.3);color:#fcd34d}
.kcard{background:linear-gradient(145deg,#161b27,#1c2132);
  border:1px solid rgba(255,255,255,.06);border-radius:14px;
  padding:18px 14px 14px;text-align:center;transition:.2s}
.kcard:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3)}
.kcard .ico{font-size:1.35rem;margin-bottom:4px}
.kcard .val{font-size:1.4rem;font-weight:700;color:#f1f5f9;margin:3px 0}
.kcard .lbl{font-size:.63rem;color:#475569;text-transform:uppercase;letter-spacing:1.2px}
.kcard .sub{font-size:.73rem;color:#94a3b8;margin-top:3px}
.kcard.c1{border-top:3px solid #3b82f6}.kcard.c2{border-top:3px solid #10b981}
.kcard.c3{border-top:3px solid #8b5cf6}.kcard.c4{border-top:3px solid #f59e0b}
.kcard.c5{border-top:3px solid #ef4444}.kcard.c6{border-top:3px solid #06b6d4}
.sec{font-size:.93rem;font-weight:600;color:#e2e8f0;
  margin:20px 0 10px;padding-left:10px;border-left:3px solid #6366f1}
.sec-y{border-left-color:#f59e0b}.sec-g{border-left-color:#10b981}
.ibox{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.18);
  border-radius:10px;padding:13px 15px;color:#93c5fd;font-size:.82rem;
  margin:8px 0;line-height:1.7}
.ibox-g{background:rgba(16,185,129,.07);border-color:rgba(16,185,129,.2);color:#6ee7b7}
.ibox-y{background:rgba(245,158,11,.07);border-color:rgba(245,158,11,.2);color:#fcd34d}
.sbox{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
  border-radius:10px;padding:12px;margin-bottom:10px}
.footer{text-align:center;padding:14px 0 4px;color:#2d3748;font-size:.72rem;
  border-top:1px solid rgba(255,255,255,.04);margin-top:20px}
[data-testid="stSidebar"]{background:#0f1117!important}
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# KONSTANTA
# ═══════════════════════════════════════════════════════════════════════
EMITEN = {
    "ADRO":{"nama":"Adaro Energy",     "warna":"#6366f1"},
    "ADMR":{"nama":"Adaro Minerals",   "warna":"#10b981"},
    "ITMG":{"nama":"Indo Tambangraya", "warna":"#f59e0b"},
    "PTBA":{"nama":"Bukit Asam",       "warna":"#ef4444"},
    "MEDC":{"nama":"Medco Energi",     "warna":"#8b5cf6"},
    "AKRA":{"nama":"AKR Corporindo",   "warna":"#06b6d4"},
}
WARNA = {k: v["warna"] for k,v in EMITEN.items()}

C_TGL, C_EM, C_AK   = "Tanggal", "Kode_Emiten", "Volatilitas_Aktual"
C_PS, C_PT, C_ERR   = ("Volatilitas_Pred_XGB_Sentimen_t1",
                        "Volatilitas_Pred_XGB_Teknikal_t1",
                        "Error_Abs")

def rgba(h, a):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def ld(h=320, title="", mt=32, mb=28):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=rgba("1c2132",0.9),
        font=dict(family="Inter",color="#94a3b8",size=11),
        xaxis=dict(gridcolor=rgba("ffffff",0.04),linecolor=rgba("ffffff",0.07)),
        yaxis=dict(gridcolor=rgba("ffffff",0.04),linecolor=rgba("ffffff",0.07)),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=rgba("ffffff",0.07),font=dict(size=11)),
        margin=dict(t=mt,b=mb,l=48,r=16), height=h,
        **({"title":dict(text=title,font=dict(size=13,color="#e2e8f0"))} if title else {}),
    )

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_csv(paths):
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            if C_TGL in df.columns:
                df[C_TGL] = pd.to_datetime(df[C_TGL])
            return df
    return pd.DataFrame()

df_sent = load_csv(["hasil_prediksi_xgb_sentimen_t1.csv"])
df_tech = load_csv(["hasil_prediksi_xgb_teknikal_t1.csv"])
df_es   = load_csv(["evaluasi_xgb_sentimen_t1.csv"])
df_et   = load_csv(["evaluasi_xgb_teknikal_t1.csv"])
df_raw  = load_csv(["dataset_merger_covid2020.csv"])

SENT_OK = len(df_sent) > 0
TECH_OK = len(df_tech) > 0

# ═══════════════════════════════════════════════════════════════════════
# METRIK
# ═══════════════════════════════════════════════════════════════════════
def calc(df, cp):
    if df.empty or C_AK not in df.columns or cp not in df.columns:
        return 0.,0.,0.
    a,p = df[C_AK].values, df[cp].values
    rmse = float(np.sqrt(np.mean((a-p)**2)))
    mae  = float(np.mean(np.abs(a-p)))
    sst  = np.sum((a-a.mean())**2)
    r2   = float(1-np.sum((a-p)**2)/sst) if sst>0 else 0.
    return rmse,mae,r2

def gm(df,ks,d=0.):
    for k in ks:
        if k in df.columns:
            try: return float(df[k].iloc[0])
            except: pass
    return d

RMSE_S,MAE_S,R2_S = (gm(df_es,['RMSE','rmse']),gm(df_es,['MAE','mae']),gm(df_es,['R2','r2'])) if not df_es.empty else calc(df_sent,C_PS)
RMSE_T,MAE_T,R2_T = (gm(df_et,['RMSE','rmse']),gm(df_et,['MAE','mae']),gm(df_et,['R2','r2'])) if not df_et.empty else calc(df_tech,C_PT)
if RMSE_S==0 and SENT_OK: RMSE_S,MAE_S,R2_S = calc(df_sent,C_PS)
if RMSE_T==0 and TECH_OK: RMSE_T,MAE_T,R2_T = calc(df_tech,C_PT)

MODEL_BEST = "XGBoost + Sentimen" if RMSE_S<=RMSE_T else "XGBoost Teknikal"
RMSE_BEST  = min(RMSE_S,RMSE_T)
EMITEN_ADA = sorted(df_sent[C_EM].unique().tolist()) if SENT_OK and C_EM in df_sent.columns else list(EMITEN.keys())

def filt(df,em_list):
    if df.empty or C_EM not in df.columns: return df
    return df[df[C_EM].isin(em_list)].copy()

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:6px 0 14px">
      <div style="font-size:2rem">📈</div>
      <div style="font-size:.93rem;font-weight:700;color:#e2e8f0">Volatilitas LQ45 Energi</div>
      <div style="font-size:.68rem;color:#475569;margin-top:2px">XGBoost · Teknikal & Sentimen</div>
    </div>""", unsafe_allow_html=True)

    menu = st.radio("Menu",[
        "🏠 Overview",
        "📉 Harga & Volatilitas",
        "🤖 Prediksi XGBoost",
        "📰 Analisis Sentimen",
        "📊 Evaluasi Model",
        "🗂 Data",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sbox">',unsafe_allow_html=True)
    st.markdown("**⚙️ Filter Emiten**")
    emiten_sel = st.multiselect("Emiten",EMITEN_ADA,default=EMITEN_ADA,label_visibility="collapsed")
    if not emiten_sel: emiten_sel = EMITEN_ADA
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="sbox">',unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.75rem;color:#94a3b8;line-height:2">
    🔹 <b style="color:#e2e8f0">Model:</b> XGBoost<br>
    🔹 <b style="color:#e2e8f0">Fitur 1:</b> 12 Indikator Teknikal<br>
    🔹 <b style="color:#e2e8f0">Fitur 2:</b> Skor Sentimen NLP<br>
    🔹 <b style="color:#e2e8f0">Target:</b> Volatilitas Return<br>
    🔹 <b style="color:#e2e8f0">Dataset:</b> COVID-19 LQ45<br>
    🔹 <b style="color:#e2e8f0">Split:</b> 80% / 20%
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="sbox">',unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.75rem;color:#94a3b8;line-height:1.9">
    👩‍🎓 <b style="color:#e2e8f0">Peneliti</b><br>
    Manajemen Teknologi<br>SIMT — ITS Surabaya<br>
    <span style="color:#6366f1;font-weight:700">2026</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
if menu == "🏠 Overview":

    # ── Data filter untuk overview
    dfs_f = filt(df_sent, emiten_sel)
    dft_f = filt(df_tech, emiten_sel)

    # Pilih emiten untuk KPI utama (1 emiten)
    col_hdr, _ = st.columns([3,4])
    with col_hdr:
        em_kpi = st.selectbox("Pilih Saham untuk KPI", emiten_sel, key="kpi_em")

    dfs_kpi = filt(df_sent,[em_kpi]).sort_values(C_TGL)
    dft_kpi = filt(df_tech,[em_kpi]).sort_values(C_TGL)
    vol_now  = dfs_kpi[C_AK].iloc[-1] if not dfs_kpi.empty else 0
    vol_prev = dfs_kpi[C_AK].iloc[-2] if len(dfs_kpi)>1 else vol_now
    vol_pct  = (vol_now-vol_prev)/abs(vol_prev)*100 if vol_prev!=0 else 0
    pred_s_v = dfs_kpi[C_PS].iloc[-1] if not dfs_kpi.empty and C_PS in dfs_kpi.columns else 0
    pred_t_v = dft_kpi[C_PT].iloc[-1] if not dft_kpi.empty and C_PT in dft_kpi.columns else 0
    n_obs    = len(dfs_kpi)
    em_nama  = EMITEN.get(em_kpi,{}).get("nama",em_kpi)
    wn_kpi   = WARNA.get(em_kpi,"#6366f1")

    # ── HERO: Logo + Judul besar bergaya gambar referensi
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b27 50%,#1a1033 100%);
                border:1px solid rgba(99,102,241,.25);border-radius:20px;
                padding:32px 36px 28px;margin-bottom:24px;
                box-shadow:0 0 50px rgba(99,102,241,.12)">

      <!-- Logo + Judul -->
      <div style="display:flex;align-items:center;gap:20px;margin-bottom:22px">
        <div style="width:68px;height:68px;border-radius:50%;
                    background:linear-gradient(135deg,#f59e0b,#ef4444);
                    display:flex;align-items:center;justify-content:center;
                    font-size:28px;box-shadow:0 4px 16px rgba(245,158,11,.3);
                    flex-shrink:0">🎓</div>
        <div>
          <div style="font-size:1.9rem;font-weight:800;color:#f1f5f9;
                      letter-spacing:-.5px;line-height:1.1">
            Dashboard Analisis<br>Volatilitas Saham
          </div>
          <div style="font-size:.85rem;color:#64748b;margin-top:6px">
            Perbandingan Model XGBoost Teknikal &amp; Sentimen &nbsp;·&nbsp;
            LQ45 Sektor Energi &nbsp;·&nbsp; Periode COVID-19 (2020–2025)
          </div>
          <div style="margin-top:10px">
            <span class="badge">XGBoost</span>
            <span class="badge badge-g">6 Emiten Energi</span>
            <span class="badge badge-y">Teknikal & Sentimen</span>
            <span class="badge">SIMT · ITS Surabaya · 2026</span>
            <span class="badge">{'✅ Data OK' if SENT_OK else '⚠️ Cek CSV'}</span>
          </div>
        </div>
      </div>

      <!-- 3 KPI besar bergaya gambar referensi -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">

        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                    border-radius:14px;padding:18px 20px">
          <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:6px">
            Volatilitas Terakhir ({em_kpi})
          </div>
          <div style="font-size:2.2rem;font-weight:800;color:#f1f5f9;
                      letter-spacing:-.5px;font-family:'Inter',sans-serif">
            {vol_now:.5f}
          </div>
          <div style="display:inline-flex;align-items:center;gap:5px;margin-top:6px;
                      background:{'rgba(239,68,68,.15)' if vol_pct>=0 else 'rgba(16,185,129,.15)'};
                      border:1px solid {'rgba(239,68,68,.3)' if vol_pct>=0 else 'rgba(16,185,129,.3)'};
                      border-radius:20px;padding:3px 10px;font-size:.75rem;font-weight:600;
                      color:{'#fca5a5' if vol_pct>=0 else '#6ee7b7'}">
            {'↑' if vol_pct>=0 else '↓'} {abs(vol_pct):.4f} ({abs(vol_pct):.2f}%)
          </div>
        </div>

        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                    border-radius:14px;padding:18px 20px">
          <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:6px">Volatilitas Rata-rata</div>
          <div style="font-size:2.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-.5px">
            {dfs_kpi[C_AK].mean():.5f}
          </div>
          <div style="font-size:.78rem;color:#94a3b8;margin-top:6px">
            Std: {dfs_kpi[C_AK].std():.5f} &nbsp;·&nbsp; n={n_obs:,}
          </div>
        </div>

        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                    border-radius:14px;padding:18px 20px">
          <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:6px">Window Volatilitas</div>
          <div style="font-size:2.2rem;font-weight:800;color:#f1f5f9;letter-spacing:-.5px">
            20 Hari
          </div>
          <div style="font-size:.78rem;color:#94a3b8;margin-top:6px">
            Rolling window · Log Return
          </div>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mini chart: Volatilitas Aktual + Prediksi side by side (seperti gambar referensi)
    ch_l, ch_r = st.columns(2)
    with ch_l:
        st.markdown('<div class="sec" style="margin-top:4px">📉 Volatilitas Aktual</div>', unsafe_allow_html=True)
        if not dfs_kpi.empty:
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(
                x=dfs_kpi[C_TGL], y=dfs_kpi[C_AK],
                mode="lines", line=dict(color=wn_kpi, width=2),
                fill="tozeroy", fillcolor=rgba(wn_kpi.lstrip("#"),0.08),
                hovertemplate="%{x|%d %b %Y}<br>Volatilitas: %{y:.5f}<extra></extra>"
            ))
            fig_ml.add_hline(y=dfs_kpi[C_AK].mean(), line_dash="dot",
                             line_color="#475569",
                             annotation_text=f"Mean {dfs_kpi[C_AK].mean():.5f}",
                             annotation_font_color="#64748b")
            fig_ml.update_layout(**ld(h=220,mt=16,mb=28))
            fig_ml.update_layout(showlegend=False,
                                  xaxis_title="Tanggal", yaxis_title="Volatilitas")
            st.plotly_chart(fig_ml, width='stretch')

    with ch_r:
        st.markdown('<div class="sec" style="margin-top:4px">🤖 Aktual vs Prediksi XGBoost</div>', unsafe_allow_html=True)
        if not dfs_kpi.empty and C_PS in dfs_kpi.columns:
            fig_mr = go.Figure()
            fig_mr.add_trace(go.Scatter(
                x=dfs_kpi[C_TGL], y=dfs_kpi[C_AK],
                mode="lines", name="Aktual",
                line=dict(color="#3b82f6", width=2),
                hovertemplate="%{x|%d %b %Y}<br>Aktual: %{y:.5f}<extra></extra>"
            ))
            fig_mr.add_trace(go.Scatter(
                x=dfs_kpi[C_TGL], y=dfs_kpi[C_PS],
                mode="lines", name="Pred Sentimen",
                line=dict(color="#f59e0b", width=1.8, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>Pred: %{y:.5f}<extra></extra>"
            ))
            fig_mr.update_layout(**ld(h=220,mt=16,mb=28))
            fig_mr.update_layout(
                hovermode="x unified",
                xaxis_title="Tanggal", yaxis_title="Volatilitas",
                legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1)
            )
            st.plotly_chart(fig_mr, width='stretch')

    # ── 6 KPI tambahan baris kedua
    st.markdown('<div class="sec">📊 Ringkasan Model</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis2=[
        ("c1","📰","RMSE Sentimen",f"{RMSE_S:.5f}",f"R² {R2_S:.4f}"),
        ("c2","📊","RMSE Teknikal",f"{RMSE_T:.5f}",f"R² {R2_T:.4f}"),
        ("c3","🏆","Model Terbaik",MODEL_BEST.split()[0]+"✓",f"RMSE {RMSE_BEST:.5f}"),
        ("c4","🤖","Pred Sentimen",f"{pred_s_v:.5f}","XGB Sentimen"),
        ("c5","📈","Pred Teknikal",f"{pred_t_v:.5f}","XGB Teknikal"),
        ("c6","🏢","Emiten Aktif",str(len(emiten_sel)),"LQ45 Energi"),
    ]
    for col,(cls,ico,lbl,val,sub) in zip([c1,c2,c3,c4,c5,c6],kpis2):
        with col:
            st.markdown(f"""<div class="kcard {cls}">
              <div class="ico">{ico}</div><div class="lbl">{lbl}</div>
              <div class="val">{val}</div><div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # Volatilitas rata-rata bulanan — satu dropdown emiten
    st.markdown('<div class="sec">📊 Tren Volatilitas Bulanan</div>',unsafe_allow_html=True)
    if SENT_OK:
        col_em_ov, _ = st.columns([2,5])
        with col_em_ov:
            em_ov = st.selectbox("Emiten", emiten_sel, key="ov_em")
        d_ov = dfs_f[dfs_f[C_EM]==em_ov].sort_values(C_TGL).copy() if C_EM in dfs_f.columns else dfs_f.sort_values(C_TGL)
        d_ov["Bulan"] = d_ov[C_TGL].dt.to_period("M").dt.to_timestamp()
        d_m = d_ov.groupby("Bulan")[C_AK].mean().reset_index()
        fig_ov = go.Figure()
        wn = WARNA.get(em_ov,"#6366f1")
        fig_ov.add_trace(go.Scatter(
            x=d_m["Bulan"], y=d_m[C_AK], mode="lines+markers",
            line=dict(color=wn,width=2), marker=dict(size=5,color=wn),
            fill="tozeroy", fillcolor=rgba(wn.lstrip("#"),0.08),
            hovertemplate="%{x|%b %Y}<br>Vol Rata2: %{y:.5f}<extra></extra>"
        ))
        fig_ov.add_hline(y=d_m[C_AK].mean(), line_dash="dot", line_color="#475569",
                         annotation_text=f"Mean {d_m[C_AK].mean():.5f}",
                         annotation_font_color="#64748b")
        fig_ov.update_layout(**ld(h=280,mt=20,mb=30))
        fig_ov.update_layout(xaxis_title="Bulan", yaxis_title="Volatilitas Rata-rata")
        st.plotly_chart(fig_ov, width='stretch')

    # Heatmap korelasi volatilitas antar emiten
    st.markdown('<div class="sec">🔥 Heatmap Korelasi Volatilitas Antar Emiten</div>',unsafe_allow_html=True)
    if SENT_OK and C_EM in dfs_f.columns:
        pv = dfs_f.pivot_table(index=C_TGL, columns=C_EM, values=C_AK).dropna()
        if len(pv.columns) > 1:
            corr = pv.corr().round(3)
            fig_hm = px.imshow(
                corr, text_auto=".3f",
                color_continuous_scale=[[0,"#ef4444"],[0.5,"#1e293b"],[1,"#10b981"]],
                zmin=-1, zmax=1,
                labels=dict(color="Korelasi")
            )
            fig_hm.update_traces(textfont=dict(size=13,color="white"))
            fig_hm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter",color="#94a3b8"),
                margin=dict(t=10,b=10,l=10,r=10), height=320,
                coloraxis_colorbar=dict(title="r",thickness=14,len=0.8)
            )
            st.plotly_chart(fig_hm, width='stretch')
        else:
            st.info("Pilih minimal 2 emiten untuk heatmap korelasi.")

    # Ringkasan per emiten + pie
    st.markdown('<div class="sec">📌 Ringkasan per Emiten</div>',unsafe_allow_html=True)
    cl, cr = st.columns([1.5, 0.7])
    with cl:
        rows=[]
        for em in emiten_sel:
            ds = filt(df_sent,[em]); dt = filt(df_tech,[em])
            rs,ms,r2s = calc(ds,C_PS); rt,mt_,r2t = calc(dt,C_PT)
            rows.append({"Emiten":em,
                "Vol Rata2":round(ds[C_AK].mean(),6) if not ds.empty else 0,
                "Vol Max":round(ds[C_AK].max(),6) if not ds.empty else 0,
                "RMSE Sentimen":round(rs,6),"RMSE Teknikal":round(rt,6),
                "Model Terbaik":"Sentimen" if rs<=rt else "Teknikal"})
        if rows:
            df_tbl = pd.DataFrame(rows)
            st.dataframe(df_tbl.style.background_gradient(
                subset=["RMSE Sentimen","RMSE Teknikal"],cmap="RdYlGn_r"),
                width='stretch', hide_index=True)
    with cr:
        if SENT_OK and C_EM in dfs_f.columns:
            vc = dfs_f[C_EM].value_counts().reset_index()
            vc.columns=["Emiten","N"]
            fig_pie = px.pie(vc,values="N",names="Emiten",hole=0.5,
                             color="Emiten",color_discrete_map=WARNA)
            fig_pie.update_traces(textposition="outside",textinfo="label+percent")
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter",color="#94a3b8"),
                showlegend=False,margin=dict(t=4,b=4,l=4,r=4),height=280)
            st.plotly_chart(fig_pie, width='stretch')

    st.markdown(f"""<div class="ibox ibox-g">
      🏆 <b>Model Terbaik: {MODEL_BEST}</b> &nbsp;|&nbsp;
      RMSE Sentimen: <b>{RMSE_S:.5f}</b> &nbsp;|&nbsp;
      RMSE Teknikal: <b>{RMSE_T:.5f}</b> &nbsp;|&nbsp;
      R² Terbaik: <b>{max(R2_S,R2_T):.4f}</b>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — HARGA & VOLATILITAS
# ═══════════════════════════════════════════════════════════════════════
elif menu == "📉 Harga & Volatilitas":
    st.markdown("""<div class="hero" style="padding:18px 26px">
      <div class="hero-title" style="font-size:1.3rem">📉 Harga & Volatilitas per Emiten</div>
      <div class="hero-sub">Analisis deskriptif historis data COVID-19</div>
    </div>""", unsafe_allow_html=True)

    cf1,cf2 = st.columns([2,5])
    with cf1:
        em2 = st.selectbox("Pilih Emiten", emiten_sel, key="hv_em")

    dfs2 = filt(df_sent,[em2]).sort_values(C_TGL)
    wn2  = WARNA.get(em2,"#6366f1")

    if dfs2.empty:
        st.warning("Data tidak tersedia untuk emiten ini.")
    else:
        # Subplot 2 baris: volatilitas + error
        fig2 = make_subplots(rows=2,cols=1,shared_xaxes=True,
            subplot_titles=[f"Volatilitas Aktual — {em2}","Error Absolut (Sentimen vs Teknikal)"],
            vertical_spacing=0.08, row_heights=[0.6,0.4])

        fig2.add_trace(go.Scatter(
            x=dfs2[C_TGL],y=dfs2[C_AK],name="Volatilitas Aktual",
            mode="lines",line=dict(color=wn2,width=2),
            fill="tozeroy",fillcolor=rgba(wn2.lstrip("#"),0.07),
            hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:.5f}<extra></extra>"
        ),row=1,col=1)
        # Mean line
        vm = dfs2[C_AK].mean()
        fig2.add_hline(y=vm,row=1,col=1,line_dash="dot",line_color="#475569",
                       annotation_text=f"Mean {vm:.5f}",
                       annotation_font_color="#64748b")

        if C_ERR in dfs2.columns:
            fig2.add_trace(go.Bar(
                x=dfs2[C_TGL],y=dfs2[C_ERR],name="Error Sentimen",
                marker_color=rgba("f59e0b",0.65),
                hovertemplate="%{x|%d %b %Y}<br>Error: %{y:.5f}<extra></extra>"
            ),row=2,col=1)
        dft2 = filt(df_tech,[em2]).sort_values(C_TGL)
        if not dft2.empty and C_ERR in dft2.columns:
            fig2.add_trace(go.Bar(
                x=dft2[C_TGL],y=dft2[C_ERR],name="Error Teknikal",
                marker_color=rgba("8b5cf6",0.65),
                hovertemplate="%{x|%d %b %Y}<br>Error: %{y:.5f}<extra></extra>"
            ),row=2,col=1)

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=rgba("1c2132",0.9),
            font=dict(family="Inter",color="#94a3b8",size=11),
            height=500,margin=dict(t=36,b=28,l=50,r=18),
            barmode="group",hovermode="x unified",showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",
                        yanchor="bottom",y=1.01,xanchor="right",x=1)
        )
        fig2.update_xaxes(gridcolor=rgba("ffffff",0.04))
        fig2.update_yaxes(gridcolor=rgba("ffffff",0.04))
        st.plotly_chart(fig2, width='stretch')

        # Statistik deskriptif
        st.markdown('<div class="sec">📋 Statistik Deskriptif Volatilitas</div>',unsafe_allow_html=True)
        desc = dfs2[C_AK].describe().round(6)
        col_s = st.columns(len(desc))
        icons = {"count":"🔢","mean":"📊","std":"📉","min":"⬇","25%":"📌","50%":"⚖️","75%":"📌","max":"⬆"}
        for i,(idx,val) in enumerate(desc.items()):
            with col_s[i]:
                st.markdown(f"""<div class="kcard c{(i%6)+1}" style="padding:12px 8px">
                  <div class="ico">{icons.get(idx,'📊')}</div>
                  <div class="lbl">{idx}</div>
                  <div class="val" style="font-size:1.1rem">{val:.5f if isinstance(val,float) else int(val)}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════
elif menu == "🤖 Prediksi XGBoost":
    st.markdown("""<div class="hero" style="padding:18px 26px">
      <div class="hero-title" style="font-size:1.3rem">🤖 Prediksi Volatilitas XGBoost</div>
      <div class="hero-sub">Aktual vs Prediksi — pilih emiten & model</div>
    </div>""", unsafe_allow_html=True)

    cf1,cf2,cf3 = st.columns([2,2,3])
    with cf1: em3 = st.selectbox("Emiten", emiten_sel, key="pr_em")
    with cf2: model_sel = st.selectbox("Model",["Keduanya","XGB + Sentimen","XGB Teknikal"])

    dfs3 = filt(df_sent,[em3]).sort_values(C_TGL)
    dft3 = filt(df_tech,[em3]).sort_values(C_TGL)
    wn3  = WARNA.get(em3,"#6366f1")

    if dfs3.empty and dft3.empty:
        st.warning("Data tidak tersedia.")
    else:
        src = dfs3 if not dfs3.empty else dft3

        # ── Chart utama: aktual vs prediksi (1 emiten, bersih)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=src[C_TGL], y=src[C_AK], name="Volatilitas Aktual",
            mode="lines", line=dict(color=wn3,width=2.5),
            hovertemplate="%{x|%d %b %Y}<br>Aktual: %{y:.5f}<extra></extra>"
        ))
        if model_sel in ["Keduanya","XGB + Sentimen"] and C_PS in dfs3.columns:
            fig3.add_trace(go.Scatter(
                x=dfs3[C_TGL], y=dfs3[C_PS], name="Pred Sentimen",
                mode="lines", line=dict(color="#f59e0b",width=1.8,dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>Pred Sentimen: %{y:.5f}<extra></extra>"
            ))
        if model_sel in ["Keduanya","XGB Teknikal"] and C_PT in dft3.columns:
            fig3.add_trace(go.Scatter(
                x=dft3[C_TGL], y=dft3[C_PT], name="Pred Teknikal",
                mode="lines", line=dict(color="#8b5cf6",width=1.8,dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>Pred Teknikal: %{y:.5f}<extra></extra>"
            ))
        fig3.update_layout(**ld(h=360,mt=20,mb=40))
        fig3.update_layout(
            hovermode="x unified",
            xaxis_title="Tanggal", yaxis_title="Volatilitas Return",
            legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1)
        )
        st.plotly_chart(fig3, width='stretch')

        # ── Scatter aktual vs prediksi
        st.markdown('<div class="sec">🎯 Scatter: Aktual vs Prediksi</div>',unsafe_allow_html=True)
        scc1, scc2 = st.columns(2)
        for col_sc, dfp, cp, title, wsc in [
            (scc1, dfs3, C_PS, "XGB + Sentimen", "#f59e0b"),
            (scc2, dft3, C_PT, "XGB Teknikal",   "#8b5cf6"),
        ]:
            if (title=="XGB + Sentimen" and model_sel not in ["Keduanya","XGB + Sentimen"]): continue
            if (title=="XGB Teknikal"   and model_sel not in ["Keduanya","XGB Teknikal"]):   continue
            with col_sc:
                if dfp.empty or cp not in dfp.columns:
                    st.info(f"Data {title} tidak tersedia"); continue
                a_,p_ = dfp[C_AK].values, dfp[cp].values
                mn_,mx_ = min(a_.min(),p_.min()), max(a_.max(),p_.max())
                rs,ms,r2s = calc(dfp,cp)
                fsc = go.Figure()
                fsc.add_trace(go.Scatter(
                    x=a_,y=p_,mode="markers",
                    marker=dict(color=wsc,size=4,opacity=0.6),
                    hovertemplate="Aktual: %{x:.5f}<br>Pred: %{y:.5f}<extra></extra>"
                ))
                fsc.add_trace(go.Scatter(
                    x=[mn_,mx_],y=[mn_,mx_],mode="lines",
                    line=dict(color="rgba(255,255,255,.2)",dash="dash",width=1),
                    showlegend=False
                ))
                fsc.update_layout(**ld(h=280,title=f"{title} — R²={r2s:.4f} · RMSE={rs:.5f}",mt=36,mb=30))
                fsc.update_layout(xaxis_title="Aktual",yaxis_title="Prediksi")
                st.plotly_chart(fsc, width='stretch')

        # ── Distribusi error
        st.markdown('<div class="sec">📊 Distribusi Error Absolut</div>',unsafe_allow_html=True)
        dc1,dc2 = st.columns(2)
        for col_d, dfp, cp, title, wsc in [
            (dc1, dfs3, C_PS, "Sentimen", "#f59e0b"),
            (dc2, dft3, C_PT, "Teknikal", "#8b5cf6"),
        ]:
            with col_d:
                if dfp.empty or C_AK not in dfp.columns or cp not in dfp.columns:
                    continue
                err_ = np.abs(dfp[C_AK].values - dfp[cp].values)
                fhist = go.Figure(go.Histogram(
                    x=err_, nbinsx=30,
                    marker_color=rgba(wsc.lstrip("#"),0.75),
                    name=title
                ))
                fhist.add_vline(x=err_.mean(),line_dash="dash",line_color="#10b981",
                                annotation_text=f"Mean {err_.mean():.5f}",
                                annotation_font_color="#10b981")
                fhist.update_layout(**ld(h=240,title=f"Distribusi Error — {title}",mt=34,mb=28))
                fhist.update_layout(xaxis_title="Error",yaxis_title="Frekuensi",showlegend=False)
                st.plotly_chart(fhist, width='stretch')

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISIS SENTIMEN
# ═══════════════════════════════════════════════════════════════════════
elif menu == "📰 Analisis Sentimen":
    st.markdown("""<div class="hero" style="padding:18px 26px">
      <div class="hero-title" style="font-size:1.3rem">📰 Analisis Sentimen</div>
      <div class="hero-sub">WordCloud, distribusi, dan korelasi sentimen vs volatilitas</div>
    </div>""", unsafe_allow_html=True)

    # ── Cari kolom sentimen di df_raw
    sent_cols = [c for c in df_raw.columns if any(k in c.lower() for k in
                 ["sentimen","sentiment","sent","score","opini","news"])] if not df_raw.empty else []

    # ── WordCloud per emiten
    st.markdown('<div class="sec sec-y">☁️ WordCloud Sentimen per Emiten</div>',unsafe_allow_html=True)

    wc_col = None
    # Cari kolom teks yang bisa dijadikan wordcloud
    text_cols = [c for c in df_raw.columns if any(k in c.lower() for k in
                 ["teks","text","berita","headline","judul","title","kata","token"])] if not df_raw.empty else []

    if text_cols or sent_cols:
        wc_em_sel = st.selectbox("Pilih Emiten untuk WordCloud",
                                  [e for e in emiten_sel if C_EM in df_raw.columns and e in df_raw[C_EM].unique()]
                                  if not df_raw.empty and C_EM in df_raw.columns else emiten_sel,
                                  key="wc_em")
        use_col = text_cols[0] if text_cols else sent_cols[0] if sent_cols else None

        if use_col and not df_raw.empty:
            df_wc = df_raw[df_raw[C_EM]==wc_em_sel] if C_EM in df_raw.columns else df_raw
            text_data = " ".join(df_wc[use_col].dropna().astype(str).tolist())
        else:
            # Buat wordcloud dari nama kolom df_raw jika tidak ada kolom teks
            text_data = None
    else:
        wc_em_sel = emiten_sel[0] if emiten_sel else "ADRO"
        text_data = None

    # Generate WordCloud
    wcol1, wcol2 = st.columns([3,2])
    with wcol1:
        if text_data and len(text_data.strip()) > 10:
            wc_obj = WordCloud(
                width=700, height=360, background_color="#0d1117",
                colormap="plasma", max_words=100,
                collocations=False, prefer_horizontal=0.85
            ).generate(text_data)
            fig_wc, ax_wc = plt.subplots(figsize=(7,3.6))
            ax_wc.imshow(wc_obj, interpolation="bilinear")
            ax_wc.axis("off")
            fig_wc.patch.set_facecolor("#0d1117")
            buf = io.BytesIO(); fig_wc.savefig(buf,format="png",bbox_inches="tight",
                                                facecolor="#0d1117",dpi=120)
            buf.seek(0); plt.close(fig_wc)
            st.image(buf, caption=f"WordCloud Sentimen — {wc_em_sel}", use_column_width=True)
        else:
            # Fallback: WordCloud dari nama-nama kolom dan emiten (demo)
            demo_text = " ".join([
                f"{em} " * 20 + "volatilitas return saham energi covid prediksi xgboost sentimen teknikal " * 10
                for em in emiten_sel
            ])
            wc_obj = WordCloud(
                width=700,height=360,background_color="#0d1117",
                colormap="cool",max_words=80
            ).generate(demo_text)
            fig_wc,ax_wc = plt.subplots(figsize=(7,3.6))
            ax_wc.imshow(wc_obj,interpolation="bilinear"); ax_wc.axis("off")
            fig_wc.patch.set_facecolor("#0d1117")
            buf=io.BytesIO(); fig_wc.savefig(buf,format="png",bbox_inches="tight",
                                              facecolor="#0d1117",dpi=120)
            buf.seek(0); plt.close(fig_wc)
            st.image(buf, caption=f"WordCloud — {wc_em_sel} (demo dari label data)",
                     use_column_width=True)
            st.markdown('<div class="ibox ibox-y">💡 WordCloud ini menggunakan label kolom dan emiten sebagai demo. Untuk wordcloud berita nyata, tambahkan kolom <code>Teks_Berita</code> atau <code>Headline</code> di dataset.</div>', unsafe_allow_html=True)

    with wcol2:
        # Heatmap sentimen vs volatilitas
        if sent_cols and not df_raw.empty:
            sc = sent_cols[0]
            df_sv = df_raw[[sc]].copy().dropna()
            if C_EM in df_raw.columns:
                df_sv[C_EM] = df_raw[C_EM]
            if C_TGL in df_raw.columns and C_TGL in filt(df_sent,emiten_sel).columns:
                df_sv[C_TGL] = df_raw[C_TGL]
                merged = pd.merge(filt(df_sent,emiten_sel)[[C_TGL,C_EM,C_AK]],
                                  df_sv[[C_TGL,sc]],on=C_TGL,how="inner").dropna()
                if not merged.empty:
                    corr_sv = merged[[sc,C_AK]].corr().iloc[0,1]
                    st.markdown(f"""<div class="ibox" style="margin-top:0">
                      <b>📊 Korelasi Sentimen vs Volatilitas</b><br><br>
                      Kolom: <code>{sc}</code><br>
                      Korelasi Pearson: <b>{corr_sv:.4f}</b><br>
                      Arah: <b>{'Positif ↑' if corr_sv>0 else 'Negatif ↓'}</b><br><br>
                      Rata-rata sentimen: <b>{merged[sc].mean():.4f}</b><br>
                      Std sentimen: <b>{merged[sc].std():.4f}</b>
                    </div>""", unsafe_allow_html=True)
        else:
            # Info kolom yang tersedia
            if not df_raw.empty:
                st.markdown(f"""<div class="ibox ibox-y">
                  <b>Kolom tersedia di dataset:</b><br>
                  <code>{"</code>, <code>".join(df_raw.columns[:10].tolist())}</code>
                  {'...' if len(df_raw.columns)>10 else ''}<br><br>
                  Tambahkan kolom dengan nama mengandung kata <b>sentimen</b>, <b>score</b>, atau <b>teks</b>
                  untuk mengaktifkan analisis sentimen penuh.
                </div>""", unsafe_allow_html=True)

    # ── Heatmap error sentimen per emiten per bulan
    st.markdown('<div class="sec sec-y">🔥 Heatmap Error Sentimen per Emiten × Bulan</div>',unsafe_allow_html=True)
    if SENT_OK and C_EM in df_sent.columns and C_ERR in df_sent.columns:
        dfs_hm = filt(df_sent,emiten_sel).copy()
        dfs_hm["Bulan"] = dfs_hm[C_TGL].dt.to_period("M").dt.to_timestamp()
        pv_hm = dfs_hm.pivot_table(index="Bulan",columns=C_EM,values=C_ERR,aggfunc="mean")
        pv_hm.index = pv_hm.index.strftime("%b %Y")
        fig_hm2 = px.imshow(
            pv_hm.round(6), text_auto=".5f",
            color_continuous_scale=[[0,"#10b981"],[0.5,"#1e293b"],[1,"#ef4444"]],
            labels=dict(color="MAE", x="Emiten", y="Bulan")
        )
        fig_hm2.update_traces(textfont=dict(size=10))
        fig_hm2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter",color="#94a3b8"),
            height=max(300, len(pv_hm)*22+80),
            margin=dict(t=10,b=10,l=80,r=20),
            coloraxis_colorbar=dict(title="MAE",thickness=12)
        )
        st.plotly_chart(fig_hm2, width='stretch')
    else:
        st.info("Data sentimen belum tersedia untuk heatmap ini.")

    # ── Perbandingan error per emiten: bar chart sentimen vs teknikal
    st.markdown('<div class="sec sec-y">🆚 MAE per Emiten — Sentimen vs Teknikal</div>',unsafe_allow_html=True)
    rows_mae=[]
    for em in emiten_sel:
        ds=filt(df_sent,[em]); dt=filt(df_tech,[em])
        _,ms,_=calc(ds,C_PS); _,mt_,_=calc(dt,C_PT)
        rows_mae.append({"Emiten":em,"MAE Sentimen":ms,"MAE Teknikal":mt_})
    if rows_mae:
        df_mae=pd.DataFrame(rows_mae)
        fig_mae=go.Figure()
        fig_mae.add_trace(go.Bar(x=df_mae["Emiten"],y=df_mae["MAE Sentimen"],
            name="Sentimen",marker_color="#f59e0b"))
        fig_mae.add_trace(go.Bar(x=df_mae["Emiten"],y=df_mae["MAE Teknikal"],
            name="Teknikal",marker_color="#8b5cf6"))
        fig_mae.update_layout(**ld(h=280,mt=20,mb=28))
        fig_mae.update_layout(barmode="group",yaxis_title="MAE",
                               xaxis_title="Emiten",hovermode="x unified")
        st.plotly_chart(fig_mae, width='stretch')

# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════════════
elif menu == "📊 Evaluasi Model":
    st.markdown("""<div class="hero" style="padding:18px 26px">
      <div class="hero-title" style="font-size:1.3rem">📊 Evaluasi Performa Model</div>
      <div class="hero-sub">RMSE · MAE · R² — XGBoost Teknikal vs Sentimen</div>
    </div>""", unsafe_allow_html=True)

    # Metrik global
    c1,c2,c3 = st.columns(3)
    for col,lbl,rmse,mae,r2,cls in [
        (c1,"XGB + Sentimen 📰",RMSE_S,MAE_S,R2_S,"c4"),
        (c2,"XGB Teknikal 📊",  RMSE_T,MAE_T,R2_T,"c3"),
        (c3,f"{'✅' if RMSE_S<=RMSE_T else '✅'} Terbaik: {MODEL_BEST}",
         RMSE_BEST,min(MAE_S,MAE_T),max(R2_S,R2_T),"c2"),
    ]:
        with col:
            st.markdown(f"""<div class="kcard {cls}" style="text-align:left;padding:20px">
              <div style="font-size:.85rem;font-weight:700;color:#e2e8f0;margin-bottom:10px">{lbl}</div>
              <div style="font-size:.75rem;color:#94a3b8;line-height:2.3">
                RMSE: <b style="color:#f1f5f9;font-size:.95rem">{rmse:.6f}</b><br>
                MAE:&nbsp; <b style="color:#f1f5f9;font-size:.95rem">{mae:.6f}</b><br>
                R²:&nbsp;&nbsp;&nbsp; <b style="color:#f1f5f9;font-size:.95rem">{r2:.5f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

    # Tabel per emiten
    st.markdown('<div class="sec sec-g">📋 Evaluasi Per Emiten</div>',unsafe_allow_html=True)
    rows_ev=[]
    for em in emiten_sel:
        ds=filt(df_sent,[em]); dt=filt(df_tech,[em])
        rs,ms,r2s=calc(ds,C_PS); rt,mt_,r2t=calc(dt,C_PT)
        rows_ev.append({"Emiten":em,
            "RMSE_S":round(rs,6),"MAE_S":round(ms,6),"R2_S":round(r2s,5),
            "RMSE_T":round(rt,6),"MAE_T":round(mt_,6),"R2_T":round(r2t,5),
            "Terbaik":"Sentimen" if rs<=rt else "Teknikal"})
    if rows_ev:
        df_ev=pd.DataFrame(rows_ev)
        df_ev.columns=["Emiten","RMSE Sent","MAE Sent","R² Sent","RMSE Tech","MAE Tech","R² Tech","Terbaik"]
        st.dataframe(df_ev.style.background_gradient(
            subset=["RMSE Sent","RMSE Tech"],cmap="RdYlGn_r"),
            width='stretch',hide_index=True)

        # Bar RMSE + R²
        st.markdown('<div class="sec sec-g">📊 RMSE & R² per Emiten</div>',unsafe_allow_html=True)
        bc1,bc2 = st.columns(2)
        with bc1:
            fb=go.Figure()
            fb.add_trace(go.Bar(x=df_ev["Emiten"],y=df_ev["RMSE Sent"],name="RMSE Sentimen",marker_color="#f59e0b"))
            fb.add_trace(go.Bar(x=df_ev["Emiten"],y=df_ev["RMSE Tech"],name="RMSE Teknikal",marker_color="#8b5cf6"))
            fb.update_layout(**ld(h=270,title="RMSE per Emiten",mt=36,mb=28))
            fb.update_layout(barmode="group",xaxis_title="Emiten",yaxis_title="RMSE",hovermode="x unified")
            st.plotly_chart(fb, width='stretch')
        with bc2:
            fr=go.Figure()
            fr.add_trace(go.Bar(x=df_ev["Emiten"],y=df_ev["R² Sent"],name="R² Sentimen",marker_color="#f59e0b"))
            fr.add_trace(go.Bar(x=df_ev["Emiten"],y=df_ev["R² Tech"],name="R² Teknikal",marker_color="#8b5cf6"))
            fr.update_layout(**ld(h=270,title="R² per Emiten",mt=36,mb=28))
            fr.update_layout(barmode="group",xaxis_title="Emiten",yaxis_title="R²",hovermode="x unified")
            st.plotly_chart(fr, width='stretch')

        # Radar
        st.markdown('<div class="sec sec-g">🕸 Radar Perbandingan Model</div>',unsafe_allow_html=True)
        rc1,rc2 = st.columns([3,2])
        with rc1:
            cats=["RMSE","MAE","1−R²"]
            vs=[RMSE_S,MAE_S,max(0,1-R2_S)]
            vt=[RMSE_T,MAE_T,max(0,1-R2_T)]
            mx_=[max(a,b)+1e-9 for a,b in zip(vs,vt)]
            vs_n=[v/m for v,m in zip(vs,mx_)]; vt_n=[v/m for v,m in zip(vt,mx_)]
            fr2=go.Figure()
            for vals,nm,clr in [(vs_n,"XGB Sentimen","#f59e0b"),(vt_n,"XGB Teknikal","#8b5cf6")]:
                fr2.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",name=nm,
                    fillcolor=rgba(clr.lstrip("#"),0.15),
                    line=dict(color=clr,width=2)))
            fr2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(bgcolor=rgba("1c2132",0.9),
                    radialaxis=dict(gridcolor=rgba("ffffff",0.08),color="#64748b"),
                    angularaxis=dict(gridcolor=rgba("ffffff",0.08),color="#94a3b8")),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=12)),
                margin=dict(t=30,b=30,l=40,r=40),height=320)
            st.plotly_chart(fr2, width='stretch')
        with rc2:
            st.markdown(f"""<div class="ibox ibox-g" style="margin-top:20px">
              🏆 <b>Kesimpulan</b><br><br>
              Model <b>{MODEL_BEST}</b> unggul dengan:<br>
              RMSE: <b>{RMSE_BEST:.6f}</b><br>
              MAE: <b>{min(MAE_S,MAE_T):.6f}</b><br>
              R²: <b>{max(R2_S,R2_T):.5f}</b><br><br>
              {'Fitur sentimen meningkatkan akurasi prediksi volatilitas dibanding teknikal saja.' if RMSE_S<=RMSE_T else 'Fitur teknikal memberikan prediksi lebih akurat pada dataset ini.'}
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 6 — DATA
# ═══════════════════════════════════════════════════════════════════════
elif menu == "🗂 Data":
    st.markdown("""<div class="hero" style="padding:18px 26px">
      <div class="hero-title" style="font-size:1.3rem">🗂 Eksplorasi Data</div>
      <div class="hero-sub">Tabel prediksi, evaluasi, dan dataset asli</div>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["📰 Prediksi Sentimen","📊 Prediksi Teknikal","📋 Evaluasi","🗃 Dataset"])
    with t1:
        df_s_show = filt(df_sent,emiten_sel) if SENT_OK else pd.DataFrame()
        if not df_s_show.empty:
            st.caption(f"**{len(df_s_show):,} baris** · Prediksi XGB Sentimen")
            st.dataframe(df_s_show, width='stretch')
            st.download_button("📥 Download",data=df_s_show.to_csv(index=False),
                file_name="pred_xgb_sentimen.csv",mime="text/csv")
        else: st.warning("File tidak ditemukan.")
    with t2:
        df_t_show = filt(df_tech,emiten_sel) if TECH_OK else pd.DataFrame()
        if not df_t_show.empty:
            st.caption(f"**{len(df_t_show):,} baris** · Prediksi XGB Teknikal")
            st.dataframe(df_t_show, width='stretch')
            st.download_button("📥 Download",data=df_t_show.to_csv(index=False),
                file_name="pred_xgb_teknikal.csv",mime="text/csv")
        else: st.warning("File tidak ditemukan.")
    with t3:
        cc1,cc2=st.columns(2)
        with cc1:
            st.caption("**Evaluasi — XGB Sentimen**")
            ev_s = df_es if not df_es.empty else pd.DataFrame({"Metrik":["RMSE","MAE","R²"],"Nilai":[RMSE_S,MAE_S,R2_S]})
            st.dataframe(ev_s, width='stretch')
        with cc2:
            st.caption("**Evaluasi — XGB Teknikal**")
            ev_t = df_et if not df_et.empty else pd.DataFrame({"Metrik":["RMSE","MAE","R²"],"Nilai":[RMSE_T,MAE_T,R2_T]})
            st.dataframe(ev_t, width='stretch')
    with t4:
        if not df_raw.empty:
            st.caption(f"**{len(df_raw):,} baris** · Dataset COVID-19")
            st.dataframe(df_raw.tail(200), width='stretch')
            st.download_button("📥 Download",data=df_raw.to_csv(index=False),
                file_name="dataset_merger_covid2020.csv",mime="text/csv")
        else: st.warning("File dataset tidak ditemukan.")

# FOOTER
st.markdown("""<div class="footer">
  📈 XGBoost StockVision · Volatilitas Saham Energi LQ45 · Teknikal &amp; Sentimen · © 2026
</div>""", unsafe_allow_html=True)
