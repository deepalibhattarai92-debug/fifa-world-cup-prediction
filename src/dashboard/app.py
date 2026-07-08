"""
FIFA World Cup 2026 Prediction Dashboard — Sports Edition v2
Run from project root: streamlit run src/dashboard/app.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED  = ROOT / "data" / "processed"
RAW        = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
EVAL_DIR   = PROCESSED / "eval"

st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Sports / FIFA 2026 theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* animations */
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes pulseGlow {
  0%,100% { box-shadow:0 0 12px rgba(255,215,0,0.25); }
  50%     { box-shadow:0 0 28px rgba(255,215,0,0.65), 0 0 60px rgba(255,215,0,0.2); }
}
@keyframes shimmer {
  0%  { background-position:-400px 0; }
  100%{ background-position:400px 0; }
}
@keyframes spin {
  from { transform:rotateY(0deg); }
  to   { transform:rotateY(360deg); }
}
@keyframes blink {
  0%,100% { opacity:1; }
  50%     { opacity:0.4; }
}

.anim-fadein { animation: fadeInUp .5s ease both; }
.anim-pulse  { animation: pulseGlow 2.2s ease-in-out infinite; }
.anim-blink  { animation: blink 1.4s ease-in-out infinite; }

/* app bg */
.stApp { background: #080d1a; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#080d1a 0%,#0b1525 60%,#081208 100%);
  border-right: 1px solid #1c3060;
}
section[data-testid="stSidebar"] * { color:#d0dce8 !important; }

/* FIFA 2026 top banner */
.wc-banner {
  background: linear-gradient(135deg,#002868 0%,#7b1c1c 60%,#EE3124 100%);
  border-radius:14px; padding:14px 22px;
  display:flex; align-items:center; gap:16px;
  margin-bottom:16px;
  animation: fadeInUp .5s ease;
}
.wc-title { font-size:1.15rem; font-weight:900; color:#fff; letter-spacing:.04em; }
.wc-sub   { font-size:.72rem;  color:rgba(255,255,255,.7); }

/* KPI cards */
.kpi {
  background: linear-gradient(135deg,#0f1628 0%,#182040 100%);
  border-radius:12px; padding:18px 16px 14px;
  border:1px solid #1c3060;
  animation: fadeInUp .4s ease both;
  position:relative; overflow:hidden;
}
.kpi::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.kpi-g::before  { background:linear-gradient(90deg,#00c853,#69f0ae); }
.kpi-b::before  { background:linear-gradient(90deg,#1565c0,#42a5f5); }
.kpi-au::before { background:linear-gradient(90deg,#f9a825,#ffd700); }
.kpi-p::before  { background:linear-gradient(90deg,#6a1b9a,#ab47bc); }
.kpi-t::before  { background:linear-gradient(90deg,#00838f,#26c6da); }
.kpi-v  { font-size:2rem; font-weight:800; color:#fff; margin:6px 0 2px; line-height:1; }
.kpi-l  { font-size:.7rem; font-weight:700; color:#7a8fa0; text-transform:uppercase; letter-spacing:.1em; }
.kpi-dp { font-size:.75rem; color:#00c853; font-weight:600; }
.kpi-dn { font-size:.75rem; color:#ff5252; font-weight:600; }
.kpi-s  { font-size:.72rem; color:#5a6a7a; }

/* section header */
.sh { font-size:.68rem; font-weight:700; color:#00c853; text-transform:uppercase;
  letter-spacing:.12em; border-bottom:1px solid #1c3060;
  padding-bottom:7px; margin-bottom:12px; }

/* content card */
.cc {
  background:linear-gradient(135deg,#0f1628 0%,#121d32 100%);
  border-radius:12px; padding:18px;
  border:1px solid #1c3060; margin-bottom:10px;
  animation: fadeInUp .45s ease;
}

/* bracket */
.bm {
  background:#111d30; border-radius:8px; padding:9px 12px;
  border-left:3px solid #1c3a7a; margin:4px 0; font-size:.82rem;
  display:flex; align-items:center; gap:6px;
}
.bm-ok { border-left:3px solid #00c853; background:linear-gradient(90deg,#081a0a,#111d30); }
.bm-live { border-left:3px solid #EE3124; }
.bm-pending { border-left:3px solid #f9a825; background:linear-gradient(90deg,#141008,#111d30); }
.bm-date { font-size:.68rem; color:#4a6080; margin-left:auto; white-space:nowrap; }
.bm-out { color:#4a5568; text-decoration:line-through; opacity:.5; }
.bm-win { color:#00c853; font-weight:700; }
.bm-tbd { color:#8a9bb0; font-size:.78rem; font-style:italic; }
.bm-sub { font-size:.65rem; color:#4a6080; text-transform:uppercase; letter-spacing:.08em; margin:10px 0 4px; }

/* podium */
.pod {
  border-radius:12px; padding:18px 12px; text-align:center;
  animation: fadeInUp .5s ease both;
}
.pod-1 { background:linear-gradient(135deg,#1a1400,#2a2000); border:1.5px solid #ffd700; animation: pulseGlow 2.2s ease-in-out infinite; }
.pod-2 { background:linear-gradient(135deg,#141414,#1e1e1e); border:1px solid #aaaaaa; }
.pod-3 { background:linear-gradient(135deg,#160e00,#1e1400); border:1px solid #cd7f32; }
.pod-rank  { font-size:1.6rem; }
.pod-name  { font-size:1rem; font-weight:800; color:#fff; margin:4px 0; }
.pod-pct   { font-size:1.4rem; font-weight:900; }
.pod-ci    { font-size:.7rem; color:#667788; }

/* why */
.wc { background:linear-gradient(135deg,#081a0a,#0f1f10); border-radius:10px;
  padding:11px 13px; border-left:4px solid #00c853; margin:5px 0; }
.wt { font-weight:700; color:#e0ead8; font-size:.87rem; }
.wd { color:#6a8a70; font-size:.76rem; margin-top:2px; }

/* match predictor cards */
.mc { border-radius:12px; padding:16px; text-align:center; border:1.5px solid; }

/* pipeline */
.ps { background:linear-gradient(135deg,#0f1628,#121d32); border-radius:12px;
  padding:16px 20px; border-left:5px solid; margin-bottom:4px; }

/* footer */
.ft { font-size:.68rem; color:#3a4a5a; text-align:center;
  padding:18px 0 6px; border-top:1px solid #1a2030; margin-top:18px; }

/* github btn override */
.stLinkButton a {
  background:linear-gradient(135deg,#1c2a4a,#0f1f3a) !important;
  color:#7ab3f5 !important; border:1px solid #3060a0 !important;
  border-radius:8px !important; font-weight:700 !important;
}

/* tabs */
.stTabs [data-baseweb="tab"] { color:#7a8fa0; }
.stTabs [aria-selected="true"] { color:#00c853 !important; border-bottom-color:#00c853 !important; }

/* radio in picker */
div[data-testid="stRadio"] label { font-size:.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#080d1a", paper_bgcolor="#080d1a",
    font=dict(color="#b0bfcc", family="Inter, sans-serif"),
)

FLAGS = {
    "France":"🇫🇷","Morocco":"🇲🇦","Spain":"🇪🇸","Argentina":"🇦🇷",
    "Portugal":"🇵🇹","Brazil":"🇧🇷","Belgium":"🇧🇪","Mexico":"🇲🇽",
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Colombia":"🇨🇴","USA":"🇺🇸","Switzerland":"🇨🇭",
    "Norway":"🇳🇴","Egypt":"🇪🇬","Paraguay":"🇵🇾","Canada":"🇨🇦",
}
CONF_COLORS = {
    "UEFA":"#1565c0","CONMEBOL":"#00838f","CONCACAF":"#e65100",
    "CAF":"#558b2f","AFC":"#6a1b9a","OFC":"#4e342e",
}
FIXTURE_TO_RESULTS_NAME = {"USA":"United States"}
FIXTURE_TO_WC_NAME = {
    "Czechia":"Czech Republic","Korea Republic":"South Korea",
    "USA":"United States","IR Iran":"Iran",
    "Türkiye":"Turkey","Côte d'Ivoire":"Ivory Coast","Congo DR":"DR Congo",
}
WC_DEFAULTS = {"appearances":0,"titles":0,"best_finish":8}
FEATURE_COLS = [
    "home_form_win_pct","home_form_goals_scored","home_form_goals_conceded",
    "home_form_goal_diff","home_form_clean_sheet_pct",
    "away_form_win_pct","away_form_goals_scored","away_form_goals_conceded",
    "away_form_goal_diff","away_form_clean_sheet_pct",
    "home_fifa_rank","home_fifa_points","away_fifa_rank","away_fifa_points",
    "home_elo_rating","away_elo_rating",
    "home_wc_appearances","home_wc_titles","home_wc_best_finish",
    "away_wc_appearances","away_wc_titles","away_wc_best_finish",
    "rank_diff","points_diff","elo_diff",
    "same_conf","neutral","match_importance","h2h_win_rate","h2h_goal_diff",
]
V1_SIM = pd.DataFrame({
    "team":["France","Morocco","Spain","Argentina","Portugal","Brazil","Belgium","Mexico","England","Colombia","USA","Switzerland","Norway","Egypt"],
    "win_pct_v1":[21.8,7.6,13.4,11.3,7.0,13.5,7.1,4.8,7.1,2.2,1.3,1.2,0.9,0.9],
})


@st.cache_data
def load_simulation():
    df = pd.read_csv(PROCESSED/"simulation_results.csv")
    n=10_000
    df["ci_low"]=(df["win_pct"]-1.96*np.sqrt(df["win_pct"]/100*(1-df["win_pct"]/100)/n)*100).round(1)
    df["ci_high"]=(df["win_pct"]+1.96*np.sqrt(df["win_pct"]/100*(1-df["win_pct"]/100)/n)*100).round(1)
    df["ci_str"]=df.apply(lambda r:f"{r.ci_low:.1f}%–{r.ci_high:.1f}%",axis=1)
    return df

@st.cache_data
def load_bracket():
    """Derive the live knockout bracket from the fixtures feed (single source of truth)."""
    from src.simulation.simulate_tournament_v2 import (
        BRACKET_2026, MATCH_ORDER, _resolve_source, derive_bracket_state,
    )
    fx = pd.read_csv(PROCESSED/"world_cup_fixtures.csv").copy()
    real = derive_bracket_state(fx)

    fx["_date"] = pd.to_datetime(fx["match_date"], errors="coerce")
    date_by_pair = {
        frozenset((str(r["home_team"]), str(r["away_team"]))): r["_date"]
        for _, r in fx.iterrows()
    }

    def match_date(a, b):
        d = date_by_pair.get(frozenset((a, b))) if (a and b) else None
        return d.strftime("%b %-d") if (d is not None and pd.notna(d)) else ""

    matches = {}
    for mid in MATCH_ORDER:
        a = _resolve_source(BRACKET_2026[mid][0], real)
        b = _resolve_source(BRACKET_2026[mid][1], real)
        matches[mid] = {
            "a": a, "b": b,
            "winner": real.get(mid),
            "played": mid in real,
            "date": match_date(a, b),
        }

    eliminated = set()
    for mid, winner in real.items():
        a = _resolve_source(BRACKET_2026[mid][0], real)
        b = _resolve_source(BRACKET_2026[mid][1], real)
        if a and b and winner:
            eliminated.add(a if winner == b else b)

    def possible_teams(source):
        """Alive teams that could still fill an unplayed bracket slot."""
        if isinstance(source, str):
            return [] if source in eliminated else [source]
        _, mid = source
        if mid in real:
            return [real[mid]]
        return possible_teams(BRACKET_2026[mid][0]) + possible_teams(BRACKET_2026[mid][1])

    def pending_slot_label(source):
        teams = possible_teams(source)
        if not teams:
            return "TBD"
        if len(teams) == 1:
            return teams[0]
        return " / ".join(teams)

    for mid in MATCH_ORDER:
        m = matches[mid]
        m["a_pending"] = pending_slot_label(BRACKET_2026[mid][0]) if not m["a"] else None
        m["b_pending"] = pending_slot_label(BRACKET_2026[mid][1]) if not m["b"] else None

    collected = pd.to_datetime(fx["collected_at"], errors="coerce").max()
    date_str = collected.strftime("%B %-d, %Y") if pd.notna(collected) else "—"

    r16 = sum(f"R16_{i}" in real for i in range(1, 9))
    qf  = sum(f"QF_{i}" in real for i in range(1, 5))
    sf  = int("SF_1" in real) + int("SF_2" in real)
    if "FINAL" in real:   status = "🏆 Champion decided"
    elif sf > 0:          status = "Semi-finals in progress"
    elif qf == 4:         status = "Quarter-finals complete"
    elif qf > 0:          status = "Quarter-finals in progress"
    elif r16 == 8:        status = "Round of 16 complete"
    elif r16 > 0:         status = f"Round of 16 · {r16}/8 played"
    else:                 status = "Knockouts starting"

    return matches, date_str, status, eliminated


@st.cache_data
def load_model_comparisons():
    return pd.read_csv(PROCESSED/"model_comparison.csv"),pd.read_csv(PROCESSED/"model_comparison_v2.csv")

@st.cache_data
def load_eval():
    return (pd.read_csv(EVAL_DIR/"confusion_matrix.csv",index_col=0),
            pd.read_csv(EVAL_DIR/"calibration.csv"),
            pd.read_csv(EVAL_DIR/"roc_curves.csv"),
            pd.read_csv(EVAL_DIR/"feature_importance.csv"),
            pd.read_csv(EVAL_DIR/"temporal_accuracy.csv"),
            pd.read_csv(EVAL_DIR/"confederation_accuracy.csv"))

@st.cache_data
def load_features():
    return pd.read_csv(PROCESSED/"features_v2.csv",parse_dates=["date"])

@st.cache_data
def load_lookup_data():
    return (pd.read_csv(PROCESSED/"fifa_rankings.csv"),
            pd.read_csv(PROCESSED/"elo_ratings.csv"),
            pd.read_csv(PROCESSED/"world_cup_history.csv"),
            pd.read_csv(RAW/"elo_code_map_v2.csv"))

@st.cache_resource
def load_model():
    with open(MODELS_DIR/"best_model_v2.pkl","rb") as f:
        p=pickle.load(f)
    return p["pipeline"],p["label_encoder"]


def build_team_lookup(teams):
    fifa,elo,wc,elo_map=load_lookup_data()
    features=load_features()
    em=elo_map.merge(elo[["country_code","elo_rating"]],left_on="elo_code",right_on="country_code",how="left")
    elo_lkp=dict(zip(em["team"],em["elo_rating"]))
    fifa_lkp=fifa.set_index("team")[["fifa_rank","fifa_points","confederation"]].to_dict("index")
    wc2=wc.copy()
    for fn,wn in FIXTURE_TO_WC_NAME.items():
        wc2.loc[wc2["team"]==wn,"team"]=fn
    wc_lkp=wc2.set_index("team")[["appearances","titles","best_finish"]].to_dict("index")
    fc=["form_win_pct","form_goals_scored","form_goals_conceded","form_goal_diff","form_clean_sheet_pct"]
    tl={}
    for team in teams:
        rn=FIXTURE_TO_RESULTS_NAME.get(team,team)
        hr=features[features["home_team"]==rn].sort_values("date")
        ar=features[features["away_team"]==rn].sort_values("date")
        form={}
        if len(hr) and pd.notna(hr.iloc[-1].get("home_form_win_pct")):
            form={c:hr.iloc[-1][f"home_{c}"] for c in fc}
        elif len(ar) and pd.notna(ar.iloc[-1].get("away_form_win_pct")):
            form={c:ar.iloc[-1][f"away_{c}"] for c in fc}
        if not form:
            form={"form_win_pct":.5,"form_goals_scored":1.5,"form_goals_conceded":1.5,"form_goal_diff":0.0,"form_clean_sheet_pct":.3}
        fd=fifa_lkp.get(team,{})
        wd=wc_lkp.get(team,WC_DEFAULTS)
        tl[team]={**{f:form[f] for f in fc},
            "fifa_rank":fd.get("fifa_rank",100),"fifa_points":fd.get("fifa_points",1000),
            "confederation":fd.get("confederation",""),
            "elo_rating":elo_lkp.get(team,1700),
            "wc_appearances":wd.get("appearances",0),"wc_titles":wd.get("titles",0),
            "wc_best_finish":wd.get("best_finish",8)}
    return tl


def build_match_row(home,away,tl):
    h,a=tl[home],tl[away]
    row={
        "home_form_win_pct":h["form_win_pct"],"home_form_goals_scored":h["form_goals_scored"],
        "home_form_goals_conceded":h["form_goals_conceded"],"home_form_goal_diff":h["form_goal_diff"],
        "home_form_clean_sheet_pct":h["form_clean_sheet_pct"],
        "away_form_win_pct":a["form_win_pct"],"away_form_goals_scored":a["form_goals_scored"],
        "away_form_goals_conceded":a["form_goals_conceded"],"away_form_goal_diff":a["form_goal_diff"],
        "away_form_clean_sheet_pct":a["form_clean_sheet_pct"],
        "home_fifa_rank":h["fifa_rank"],"home_fifa_points":h["fifa_points"],
        "away_fifa_rank":a["fifa_rank"],"away_fifa_points":a["fifa_points"],
        "home_elo_rating":h["elo_rating"],"away_elo_rating":a["elo_rating"],
        "home_wc_appearances":h["wc_appearances"],"home_wc_titles":h["wc_titles"],
        "home_wc_best_finish":h["wc_best_finish"],
        "away_wc_appearances":a["wc_appearances"],"away_wc_titles":a["wc_titles"],
        "away_wc_best_finish":a["wc_best_finish"],
        "rank_diff":h["fifa_rank"]-a["fifa_rank"],"points_diff":h["fifa_points"]-a["fifa_points"],
        "elo_diff":h["elo_rating"]-a["elo_rating"],
        "same_conf":int(h["confederation"]==a["confederation"]),
        "neutral":True,"match_importance":5,"h2h_win_rate":np.nan,"h2h_goal_diff":np.nan,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def quick_sim(qf_bracket, n=2000):
    """Run a fast simulation given 4 QF matchups [(t1,t2),(t3,t4),(t5,t6),(t7,t8)]."""
    all_teams=list(set(t for pair in qf_bracket for t in pair))
    pipeline,label_encoder=load_model()
    tl=build_team_lookup(all_teams)
    classes=list(label_encoder.classes_)
    ai,di,hi=classes.index("away_win"),classes.index("draw"),classes.index("home_win")
    wp={}
    for home in all_teams:
        for away in all_teams:
            if home==away: continue
            row=build_match_row(home,away,tl)
            p=pipeline.predict_proba(row)[0]
            ph=p[hi]+p[di]*.5; pa=p[ai]+p[di]*.5
            wp[(home,away)]=ph/(ph+pa)
    wins={t:0 for t in all_teams}
    rng=np.random.default_rng(42)
    for _ in range(n):
        sf=[]
        for h,a in qf_bracket:
            sf.append(h if rng.random()<wp[(h,a)] else a)
        final=[]
        for i in range(0,4,2):
            final.append(sf[i] if rng.random()<wp[(sf[i],sf[i+1])] else sf[i+1])
        champ=final[0] if rng.random()<wp[(final[0],final[1])] else final[1]
        wins[champ]+=1
    return sorted({t:v/n*100 for t,v in wins.items()}.items(),key=lambda x:-x[1])


def radar_chart(home,away,tl):
    h,a=tl[home],tl[away]
    fifa,_,_,_=load_lookup_data()
    mp=fifa["fifa_points"].max()
    cats=["FIFA Points","Elo Rating","Form Win %","WC Experience","Goals Scored","Clean Sheets"]
    hv=[h["fifa_points"]/mp*100,min(h["elo_rating"]/2200*100,100),h["form_win_pct"]*100,
        min(h["wc_appearances"]/22*100,100),min(h["form_goals_scored"]/3*100,100),h["form_clean_sheet_pct"]*100]
    av=[a["fifa_points"]/mp*100,min(a["elo_rating"]/2200*100,100),a["form_win_pct"]*100,
        min(a["wc_appearances"]/22*100,100),min(a["form_goals_scored"]/3*100,100),a["form_clean_sheet_pct"]*100]
    fig=go.Figure()
    for vals,name,color,fill in [(hv,home,"#00c853","rgba(0,200,83,0.15)"),(av,away,"#ff5252","rgba(255,82,82,0.15)")]:
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
            fill="toself",name=name,line=dict(color=color,width=2),fillcolor=fill))
    fig.update_layout(polar=dict(bgcolor="#080d1a",
        radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=9,color="#445566"),gridcolor="#1c2a3e",linecolor="#1c2a3e"),
        angularaxis=dict(tickfont=dict(size=10,color="#7a8fa0"),gridcolor="#1c2a3e")),
        showlegend=True,legend=dict(orientation="h",x=0.25,y=-0.12),
        height=360,plot_bgcolor="#080d1a",paper_bgcolor="#080d1a",
        font=dict(color="#b0bfcc"),margin=dict(l=20,r=20,t=20,b=50))
    return fig


def win_gauge(p_home,home_name):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=round(p_home*100,1),
        number={"suffix":"%","font":{"size":34,"color":"#00c853"}},
        title={"text":f"{FLAGS.get(home_name,'')} {home_name} Win Probability","font":{"size":13,"color":"#7a8fa0"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#445566","tickfont":{"color":"#445566"}},
               "bar":{"color":"#00c853"},"bgcolor":"#111827","borderwidth":0,
               "steps":[{"range":[0,33],"color":"#0d1520"},{"range":[33,66],"color":"#0f1a28"},{"range":[66,100],"color":"#081508"}],
               "threshold":{"line":{"color":"#ffd700","width":3},"thickness":.75,"value":50}}))
    fig.update_layout(height=240,plot_bgcolor="#080d1a",paper_bgcolor="#080d1a",
                      font=dict(color="#b0bfcc"),margin=dict(l=30,r=30,t=40,b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚽ FIFA WC 2026")
    st.markdown("##### Machine Learning Predictions")
    st.divider()
    page=st.radio("Navigate",[
        "🏠  Overview","🎯  Match Predictor","🏆  Simulation",
        "📊  Model Performance","🔧  Feature Engineering",
        "ℹ️  About"],
        label_visibility="collapsed")
    st.divider()
    sim_sb=load_simulation()
    top=sim_sb.iloc[0]
    st.markdown("**Model's Champion Pick**")
    st.markdown(f"## {FLAGS.get(top['team'],'')} {top['team']}")
    st.markdown(f"**{top['win_pct']:.1f}%** championship probability")
    st.caption(f"95% CI: {top['ci_str']}")
    st.divider()
    st.link_button("⭐ View on GitHub", "https://github.com/deepalibhattarai92-debug/fifa-world-cup-prediction")
    _bm, _bdate, _bstatus, _ = load_bracket()
    st.caption(f"Data: {_bdate} · {_bstatus}")


# ===========================================================================
# OVERVIEW
# ===========================================================================
if page=="🏠  Overview":
    matches, bracket_date, bracket_status, eliminated = load_bracket()
    # FIFA 2026 branding banner
    st.markdown(f"""<div class="wc-banner">
      <span style="font-size:2.2rem">🏆</span>
      <div>
        <div class="wc-title">FIFA WORLD CUP 2026™</div>
        <div class="wc-sub">Machine Learning Predictions · XGBoost (Tuned) · 10,000 Monte Carlo Simulations</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:.72rem;color:rgba(255,255,255,.5)">Data as of</div>
        <div style="font-size:.9rem;color:#fff;font-weight:700">{bracket_date}</div>
        <div class="anim-blink" style="font-size:.68rem;color:#EE3124">⚡ {bracket_status}</div>
      </div>
    </div>""",unsafe_allow_html=True)

    # Sports stat row — derived live from the bracket
    sim=load_simulation()
    fav=sim.iloc[0]
    r16_ids=[f"R16_{i}" for i in range(1,9)]
    qf_ids=[f"QF_{i}" for i in range(1,5)]
    r16_played=sum(matches[m]["played"] for m in r16_ids)
    qf_set=sum(1 for m in qf_ids if matches[m]["a"] and matches[m]["b"])
    alive_count=16-len(eliminated)

    k1,k2,k3,k4,k5=st.columns(5)
    for col,(lbl,val,sub,cls) in zip([k1,k2,k3,k4,k5],[
        ("Teams Remaining",f"{alive_count}","of 16 in the knockouts","kpi-g"),
        ("Round of 16",f"{r16_played} / 8","matches played","kpi-b"),
        ("Quarter-finals",f"{qf_set} / 4","matchups confirmed","kpi-au"),
        ("Title Favorite",f"{FLAGS.get(fav['team'],'')} {fav['team']}",f"{fav['win_pct']:.1f}% to lift the trophy","kpi-p"),
        ("The Final","Jul 19","MetLife Stadium, NY","kpi-t"),
    ]):
        col.markdown(f'<div class="kpi {cls}"><div class="kpi-l">{lbl}</div><div class="kpi-v">{val}</div><div class="kpi-dp">{sub}</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # Champion podium
    st.markdown('<div class="sh">🏅 Predicted Championship Podium</div>',unsafe_allow_html=True)
    pod1,pod2,pod3=st.columns(3)

    t1s,t2s,t3s=sim.iloc[0],sim.iloc[1],sim.iloc[2]
    pod1.markdown(f'<div class="pod pod-1"><div class="pod-rank">🏆</div><div style="font-size:1.6rem">{FLAGS.get(t1s["team"],"")}</div><div class="pod-name">{t1s["team"]}</div><div class="pod-pct" style="color:#ffd700">{t1s["win_pct"]:.1f}%</div><div class="pod-ci">{t1s["ci_str"]}</div><div style="color:#5a4a20;font-size:.7rem;margin-top:4px">Champion</div></div>',unsafe_allow_html=True)
    pod2.markdown(f'<div class="pod pod-2"><div class="pod-rank">🥈</div><div style="font-size:1.4rem">{FLAGS.get(t2s["team"],"")}</div><div class="pod-name">{t2s["team"]}</div><div class="pod-pct" style="color:#aaaaaa">{t2s["win_pct"]:.1f}%</div><div class="pod-ci">{t2s["ci_str"]}</div><div style="color:#555;font-size:.7rem;margin-top:4px">Runner-up</div></div>',unsafe_allow_html=True)
    pod3.markdown(f'<div class="pod pod-3"><div class="pod-rank">🥉</div><div style="font-size:1.4rem">{FLAGS.get(t3s["team"],"")}</div><div class="pod-name">{t3s["team"]}</div><div class="pod-pct" style="color:#cd7f32">{t3s["win_pct"]:.1f}%</div><div class="pod-ci">{t3s["ci_str"]}</div><div style="color:#4a3010;font-size:.7rem;margin-top:4px">2nd Runner-up</div></div>',unsafe_allow_html=True)

    st.divider()

    # BRACKET — survivors only in later rounds; losers struck through in results
    def _fl(name):
        return f'{FLAGS.get(name,"")} {name}'.strip()

    def _fmt_pending(label):
        if label == "TBD":
            return '<span class="bm-tbd">TBD</span>'
        parts = [f'<span class="bm-tbd">{_fl(t.strip())}</span>' for t in label.split(" / ")]
        return " / ".join(parts)

    def render_r16_done(col, mid):
        m=matches[mid]
        loser=m["b"] if m["winner"]==m["a"] else m["a"]
        date=f'<span class="bm-date">{m["date"]}</span>' if m["date"] else ""
        col.markdown(
            f'<div class="bm bm-ok">✅ <span class="bm-win">{_fl(m["winner"])}</span>'
            f' <span style="color:#445566">def.</span> <span class="bm-out">{_fl(loser)}</span>{date}</div>',
            unsafe_allow_html=True)

    def render_r16_upcoming(col, mid):
        m=matches[mid]
        date=f'<span class="bm-date">{m["date"]}</span>' if m["date"] else ""
        col.markdown(
            f'<div class="bm bm-live"><span class="anim-blink" style="color:#EE3124">🔴</span>'
            f' <span style="color:#c0ccd8">{_fl(m["a"])} vs {_fl(m["b"])}</span>{date}</div>',
            unsafe_allow_html=True)

    def render_knockout(col, mid):
        m=matches[mid]
        date=f'<span class="bm-date">{m["date"]}</span>' if m["date"] else ""
        if m["played"]:
            col.markdown(
                f'<div class="bm bm-ok">✅ <span class="bm-win">{_fl(m["winner"])}</span>'
                f' <span style="color:#445566">advances</span>{date}</div>',
                unsafe_allow_html=True)
        elif m["a"] and m["b"]:
            col.markdown(
                f'<div class="bm bm-pending">⚔️ <strong style="color:#ffd700">{_fl(m["a"])}</strong>'
                f' <span style="color:#445566">vs</span> <strong style="color:#ffd700">{_fl(m["b"])}</strong>{date}</div>',
                unsafe_allow_html=True)
        else:
            left=_fl(m["a"]) if m["a"] else _fmt_pending(m["a_pending"])
            right=_fl(m["b"]) if m["b"] else _fmt_pending(m["b_pending"])
            col.markdown(
                f'<div class="bm bm-pending">⏳ {left} <span style="color:#445566">vs</span> {right}</div>',
                unsafe_allow_html=True)

    r16_done=[m for m in [f"R16_{i}" for i in range(1,9)] if matches[m]["played"]]
    r16_upcoming=[m for m in [f"R16_{i}" for i in range(1,9)] if not matches[m]["played"]]

    st.markdown('<div class="sh">⚽ Knockout Bracket</div>',unsafe_allow_html=True)

    if r16_done:
        st.markdown('<div class="bm-sub">Round of 16 — Results</div>',unsafe_allow_html=True)
        d1,d2=st.columns(2)
        half=(len(r16_done)+1)//2
        for col,mids in [(d1,r16_done[:half]),(d2,r16_done[half:])]:
            for mid in mids:
                render_r16_done(col,mid)

    if r16_upcoming:
        st.markdown('<div class="bm-sub">Round of 16 — Still to play</div>',unsafe_allow_html=True)
        u1,u2=st.columns(2)
        half=(len(r16_upcoming)+1)//2
        for col,mids in [(u1,r16_upcoming[:half]),(u2,r16_upcoming[half:])]:
            for mid in mids:
                render_r16_upcoming(col,mid)

    st.markdown('<div class="bm-sub" style="margin-top:14px">Quarter-finals</div>',unsafe_allow_html=True)
    qf1,qf2=st.columns(2)
    for col,mids in [(qf1,["QF_1","QF_2"]),(qf2,["QF_3","QF_4"])]:
        for mid in mids:
            render_knockout(col,mid)
    st.markdown(f'<div class="bm" style="border-left:3px solid #f9a825;background:linear-gradient(90deg,#1a1200,#111d30);margin-top:6px">🏆 <strong style="color:#f9a825">Final</strong> — <strong style="color:#ffd700">July 19, 2026</strong> · MetLife Stadium, New York</div>',unsafe_allow_html=True)

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)


# ===========================================================================
# MATCH PREDICTOR
# ===========================================================================
elif page=="🎯  Match Predictor":
    st.markdown('<div class="wc-banner"><span style="font-size:2rem">🎯</span><div><div class="wc-title">Match Predictor</div><div class="wc-sub">Select any two international teams · XGBoost (Tuned) · V2 features</div></div></div>',unsafe_allow_html=True)

    fifa,_,_,_=load_lookup_data()
    pipeline,label_encoder=load_model()
    all_teams=sorted(fifa["team"].tolist())

    c1,vs_c,c2=st.columns([2,.4,2])
    with c1:
        ht=st.selectbox("🏠 Home / Team 1 — type to search",all_teams,
            index=all_teams.index("France") if "France" in all_teams else 0)
    with vs_c:
        st.markdown("<br><div style='text-align:center;font-size:1.8rem;font-weight:900;color:#2a3a5a;padding-top:8px'>VS</div>",unsafe_allow_html=True)
    with c2:
        at=st.selectbox("✈️ Away / Team 2 — type to search",all_teams,
            index=all_teams.index("Spain") if "Spain" in all_teams else 1)

    neutral=st.checkbox("⚽ Neutral venue (World Cup match)",value=True)
    predict_btn=st.button("🔮  Predict Match Outcome",type="primary",use_container_width=True)

    if predict_btn:
        if ht==at:
            st.warning("Please select two different teams.")
        else:
            with st.spinner("Analysing..."):
                tl=build_team_lookup(list(set([ht,at])))
                row=build_match_row(ht,at,tl)
                if not neutral: row["neutral"]=False
                probs=pipeline.predict_proba(row)[0]
                classes=list(label_encoder.classes_)
                p_away=probs[classes.index("away_win")]
                p_draw=probs[classes.index("draw")]
                p_home=probs[classes.index("home_win")]

            st.divider()
            hf,af=FLAGS.get(ht,""),FLAGS.get(at,"")
            max_p=max(p_home,p_draw,p_away)
            p1,p2,p3=st.columns(3)
            for col,(label,p,color) in zip([p1,p2,p3],[
                (f"{hf} {ht} Win",p_home,"#00c853"),
                ("Draw",p_draw,"#f9a825"),
                (f"{af} {at} Win",p_away,"#ff5252"),
            ]):
                glow=f"animation:pulseGlow 2s ease-in-out infinite;" if p==max_p else ""
                col.markdown(f'<div class="mc" style="border-color:{color};{glow}"><div style="font-size:.75rem;color:#7a8fa0;text-transform:uppercase;letter-spacing:.08em">{label}</div><div style="font-size:2.2rem;font-weight:900;color:{color};margin:8px 0">{p:.1%}</div><div style="background:{color}22;border-radius:4px;height:8px"><div style="background:{color};width:{p*100:.0f}%;height:100%;border-radius:4px"></div></div></div>',unsafe_allow_html=True)

            winner=ht if p_home==max_p else (at if p_away==max_p else "Draw")
            result_label=f"{hf} {ht}" if p_home==max_p else (f"{af} {at}" if p_away==max_p else "a Draw")
            st.success(f"**{result_label}** is favoured at **{max_p:.1%}** · Based on XGBoost (Tuned) V2")
            st.caption("Team-level features only · No player, injury or squad data")

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)


# ===========================================================================
# SIMULATION
# ===========================================================================
elif page=="🏆  Simulation":
    _sm, _sdate, _sstatus, _ = load_bracket()
    st.markdown(f'<div class="wc-banner"><span style="font-size:2rem">🏆</span><div><div class="wc-title">Tournament Simulation</div><div class="wc-sub">10,000 Monte Carlo simulations · XGBoost (Tuned) · {_sdate}</div></div></div>',unsafe_allow_html=True)

    sim=load_simulation()
    tab1,tab2,tab3=st.tabs(["🥇 Championship Odds","📈 Stage Progression","🔄 V1 vs V2 Shift"])

    with tab1:
        cl,cr=st.columns([1.15,1])
        with cl:
            bar_c=["#ffd700","#aaaaaa","#cd7f32"]+["#1565c0"]*5+["#2a3a4a"]*6
            fig_c=go.Figure(go.Bar(
                x=sim["win_pct"],y=[f"{FLAGS.get(t,'')} {t}" for t in sim["team"]],
                orientation="h",marker=dict(color=bar_c[:len(sim)],line=dict(width=0)),
                text=[f"  {p:.1f}%  ({ci})" for p,ci in zip(sim["win_pct"],sim["ci_str"])],
                textposition="outside",textfont=dict(size=11),
                hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>",
            ))
            fig_c.update_layout(height=460,xaxis=dict(title="Championship Probability (%)",range=[0,28],gridcolor="#1c2a3e"),
                                yaxis=dict(autorange="reversed",tickfont=dict(size=12)),**PLOTLY_LAYOUT)
            st.plotly_chart(fig_c,use_container_width=True)
        with cr:
            disp=sim.copy()
            disp.insert(0,"🏅",["🥇","🥈","🥉"]+[""]*(len(disp)-3))
            disp["Team"]=disp["team"].apply(lambda t:f"{FLAGS.get(t,'')} {t}")
            disp=disp.rename(columns={"qf_pct":"QF%","sf_pct":"SF%","final_pct":"Final%","win_pct":"Win%","ci_str":"95% CI"})
            st.dataframe(disp[["🏅","Team","QF%","SF%","Final%","Win%","95% CI"]],
                hide_index=True,use_container_width=True,height=460,
                column_config={"Win%":st.column_config.ProgressColumn("Win%",format="%.1f%%",min_value=0,max_value=25),
                               "QF%":st.column_config.NumberColumn(format="%.1f%%"),"SF%":st.column_config.NumberColumn(format="%.1f%%"),
                               "Final%":st.column_config.NumberColumn(format="%.1f%%")})
            st.info(f"Top 5 cumulative: **{sim.head(5)['win_pct'].sum():.1f}%**")

        # Why the model favours the top pick — reasoning from model features
        fav_team=sim.iloc[0]["team"]
        st.markdown(f'<div class="sh" style="margin-top:10px">🤔 Why {fav_team}? — Model Reasoning</div>',unsafe_allow_html=True)
        fr=build_team_lookup([fav_team]).get(fav_team,{})
        why_cols=st.columns(4)
        for col,(icon,title,detail) in zip(why_cols,[
            ("🏆","World Cup Pedigree",f"{int(fr.get('wc_titles',0))} titles · {int(fr.get('wc_appearances',0))} appearances"),
            ("📊","FIFA Ranking",f"{int(fr.get('fifa_points',0)):,} pts · #{int(fr.get('fifa_rank',0))} globally"),
            ("⚡","Elo Rating",f"{int(fr.get('elo_rating',0)):,} — among the highest remaining"),
            ("📈","Recent Form",f"{fr.get('form_win_pct',0):.0%} win rate · last 10 competitive"),
        ]):
            col.markdown(f'<div class="wc"><span style="font-size:1.1rem">{icon}</span> <span class="wt">{title}</span><div class="wd">{detail}</div></div>',unsafe_allow_html=True)
        st.caption("Model features only — no player, injury or squad data")

    with tab2:
        stages=["qf_pct","sf_pct","final_pct","win_pct"]
        slabels=["Quarter-Final","Semi-Final","Final","Champion"]
        scolors=["#1565c0","#00838f","#f9a825","#00c853"]
        fig_p=go.Figure()
        for s,sl,sc in zip(stages,slabels,scolors):
            fig_p.add_trace(go.Bar(name=sl,x=[f"{FLAGS.get(t,'')} {t}" for t in sim["team"]],
                y=sim[s],marker_color=sc,text=sim[s].apply(lambda x:f"{x:.0f}%"),textposition="auto"))
        fig_p.update_layout(barmode="group",height=400,xaxis=dict(tickangle=-30,gridcolor="#1c2a3e"),
                            yaxis=dict(title="Probability (%)",gridcolor="#1c2a3e"),
                            legend=dict(orientation="h",y=1.06,x=0),**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=30,b=80))
        st.plotly_chart(fig_p,use_container_width=True)

        st.markdown('<div class="sh">Stage Probability Heatmap</div>',unsafe_allow_html=True)
        heat=sim[["team"]+stages].set_index("team"); heat.columns=slabels
        fig_h=go.Figure(go.Heatmap(z=heat.values.T,
            x=[f"{FLAGS.get(t,'')} {t}" for t in heat.index],y=slabels,
            text=[[f"{v:.1f}%" for v in row] for row in heat.values.T],texttemplate="%{text}",
            colorscale=[[0,"#080d1a"],[0.5,"#1565c0"],[1,"#00c853"]],showscale=True))
        fig_h.update_layout(height=270,**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=60))
        st.plotly_chart(fig_h,use_container_width=True)

    with tab3:
        comp=sim[["team","win_pct"]].merge(V1_SIM,on="team")
        comp["delta"]=(comp["win_pct"]-comp["win_pct_v1"]).round(1)
        comp=comp.sort_values("delta")
        fig_d=go.Figure(go.Bar(
            x=comp["delta"],y=[f"{FLAGS.get(t,'')} {t}" for t in comp["team"]],
            orientation="h",marker=dict(color=["#00c853" if d>0 else "#ff5252" for d in comp["delta"]],line=dict(width=0)),
            text=[f"{d:+.1f}pp" for d in comp["delta"]],textposition="outside"))
        fig_d.add_vline(x=0,line_color="#445566",line_width=1.5)
        fig_d.update_layout(height=400,xaxis=dict(title="Change V2 − V1 (percentage points)",gridcolor="#1c2a3e"),**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_d,use_container_width=True)
        st.caption("Positive = V2 gives team a higher championship probability than V1. H2H and match importance features drive most shifts.")

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)


# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page=="📊  Model Performance":
    st.markdown('<div class="wc-banner"><span style="font-size:2rem">📊</span><div><div class="wc-title">Model Performance</div><div class="wc-sub">5,925 competitive test matches (2018–2026) · XGBoost (Tuned) · No data leakage</div></div></div>',unsafe_allow_html=True)

    cm,cal,roc,fi,temp,conf_acc=load_eval()
    v1c,v2c=load_model_comparisons()
    v1b=v1c.loc[v1c["log_loss"].idxmin()]; v2b=v2c.loc[v2c["log_loss"].idxmin()]

    k1,k2,k3,k4=st.columns(4)
    for col,(lbl,key,hb,cls) in zip([k1,k2,k3,k4],[
        ("Accuracy","accuracy",True,"kpi-g"),("F1 Score","f1",True,"kpi-b"),
        ("ROC-AUC","roc_auc",True,"kpi-au"),("Log Loss ↓","log_loss",False,"kpi-t"),
    ]):
        v1v,v2v=v1b[key],v2b[key]; d=v2v-v1v
        ok=(d>0) if hb else (d<0)
        delta_cls = "kpi-dp" if ok else "kpi-dn"
        arrow = "↑" if d>0 else "↓"
        col.markdown(f'<div class="kpi {cls}"><div class="kpi-l">{lbl}</div><div class="kpi-v">{v2v:.3f}</div><div class="{delta_cls}">{arrow} {abs(d):.3f} vs baseline ({v1v:.3f})</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    t_cm,t_cal,t_roc,t_time=st.tabs(["🔢 Confusion Matrix","📐 Calibration","📉 ROC","📅 Temporal & Confederation"])

    with t_cm:
        c1,c2=st.columns([1.2,1])
        with c1:
            z=cm.values.tolist(); tots=[sum(r) for r in z]
            zp=[[f"<b>{v}</b><br>({v/t:.1%})" for v,t in zip(row,tots)] for row,t in zip(z,tots)]
            fig_cm=go.Figure(go.Heatmap(
                z=[[v/t for v,t in zip(row,tots)] for row,t in zip(z,tots)],
                text=zp,texttemplate="%{text}",
                x=["Pred: Away Win","Pred: Draw","Pred: Home Win"],
                y=["Actual: Away Win","Actual: Draw","Actual: Home Win"],
                colorscale=[[0,"#080d1a"],[0.3,"#1565c0"],[1,"#00c853"]],showscale=True))
            fig_cm.update_layout(height=370,**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=20,b=20))
            st.plotly_chart(fig_cm,use_container_width=True)
        with c2:
            total=sum(sum(r) for r in z); correct=sum(z[i][i] for i in range(3))
            st.markdown(f'<div class="kpi kpi-g"><div class="kpi-l">Overall Accuracy</div><div class="kpi-v">{correct/total:.1%}</div><div class="kpi-s">{correct:,} / {total:,} correct predictions</div></div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            for icon,title,detail in [
                ("🏠","Home wins",f"{z[2][2]:,}/{tots[2]:,} correct ({z[2][2]/tots[2]:.0%}) — model strongest here"),
                ("🤝","Draws","Almost always wrong — draws are football's hardest outcome"),
                ("✈️","Away wins",f"{z[0][0]:,}/{tots[0]:,} correct ({z[0][0]/tots[0]:.0%}) — harder than home"),
                ("⚠️","Home bias","Model over-predicts home wins — inherent in football data"),
            ]:
                st.markdown(f'<div style="padding:8px 0;border-bottom:1px solid #1a2030"><span style="font-size:.95rem">{icon}</span> <strong style="color:#c0ccd8;font-size:.87rem">{title}</strong><div style="color:#5a6a7a;font-size:.78rem;margin-top:2px">{detail}</div></div>',unsafe_allow_html=True)

    with t_cal:
        c1,c2=st.columns([1.3,1])
        with c1:
            fig_cal=go.Figure()
            fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect",line=dict(dash="dash",color="#445566",width=1.5)))
            for cls,color,label in [("home_win","#00c853","Home Win"),("draw","#f9a825","Draw"),("away_win","#ff5252","Away Win")]:
                s=cal[cal["class"]==cls]
                fig_cal.add_trace(go.Scatter(x=s["mean_predicted_prob"],y=s["fraction_positive"],
                    mode="lines+markers",name=label,line=dict(color=color,width=2.5),marker=dict(size=8)))
            fig_cal.update_layout(height=360,xaxis=dict(title="Predicted Probability",gridcolor="#1c2a3e",range=[0,1]),
                                  yaxis=dict(title="Actual Frequency",gridcolor="#1c2a3e",range=[0,1]),
                                  legend=dict(x=0,y=1),**PLOTLY_LAYOUT)
            st.plotly_chart(fig_cal,use_container_width=True)
        with c2:
            st.markdown('<div class="sh">Why This Matters</div>',unsafe_allow_html=True)
            st.markdown('<div class="cc" style="font-size:.85rem;line-height:1.7;color:#b0bfcc">A <strong style="color:#fff">calibrated model</strong> is one where a predicted probability of 70% means the event actually happens 70% of the time.<br><br>This is why <strong style="color:#00c853">Log Loss</strong> was chosen as the primary metric — not accuracy.<br><br>Calibrated probabilities produce <strong style="color:#00c853">more reliable simulations</strong> since each round feeds into the next.</div>',unsafe_allow_html=True)
            for lbl,v1,v2 in [("Log Loss",0.886,0.851),("ROC-AUC",0.747,0.775)]:
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a2030;font-size:.85rem"><span style="color:#7a8fa0">{lbl}</span><span style="color:#c0ccd8">{v1} → <strong style="color:#00c853">{v2}</strong></span></div>',unsafe_allow_html=True)

    with t_roc:
        c1,c2=st.columns([1.3,1])
        with c1:
            fig_r=go.Figure()
            fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random (0.50)",line=dict(dash="dash",color="#445566",width=1.5)))
            cal_labels={"home_win":"Home Win","draw":"Draw","away_win":"Away Win"}
            _roc_fill={"home_win":"rgba(0,200,83,0.06)","draw":"rgba(249,168,37,0.06)","away_win":"rgba(255,82,82,0.06)"}
            for cls,color in [("home_win","#00c853"),("draw","#f9a825"),("away_win","#ff5252")]:
                s=roc[roc["class"]==cls]; av=s["auc"].iloc[0]
                fig_r.add_trace(go.Scatter(x=s["fpr"],y=s["tpr"],mode="lines",
                    name=f"{cal_labels[cls]} (AUC={av:.3f})",line=dict(color=color,width=2.5),
                    fill="tozeroy",fillcolor=_roc_fill[cls]))
            fig_r.update_layout(height=360,xaxis=dict(title="False Positive Rate",gridcolor="#1c2a3e",range=[0,1]),
                                yaxis=dict(title="True Positive Rate",gridcolor="#1c2a3e",range=[0,1]),
                                legend=dict(x=0.35,y=0.08),**PLOTLY_LAYOUT)
            st.plotly_chart(fig_r,use_container_width=True)
        with c2:
            st.markdown('<div class="sh">AUC per Class</div>',unsafe_allow_html=True)
            aucs={c:roc[roc["class"]==c]["auc"].iloc[0] for c in ["home_win","draw","away_win"]}
            for cls,color in [("home_win","#00c853"),("draw","#f9a825"),("away_win","#ff5252")]:
                st.markdown(f'<div class="kpi" style="border-top:3px solid {color};margin-bottom:8px"><div class="kpi-l">{cal_labels[cls]}</div><div class="kpi-v" style="color:{color}">{aucs[cls]:.3f}</div></div>',unsafe_allow_html=True)
            macro=np.mean(list(aucs.values()))
            st.markdown(f'<div class="kpi kpi-au"><div class="kpi-l">Macro AUC</div><div class="kpi-v">{macro:.3f}</div><div class="kpi-s">Average across 3 classes</div></div>',unsafe_allow_html=True)

    with t_time:
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="sh">Accuracy by Year (Test Period)</div>',unsafe_allow_html=True)
            fig_t=go.Figure()
            fig_t.add_trace(go.Bar(x=temp["year"].astype(str),y=temp["accuracy"]*100,
                marker=dict(color=["#ff5252" if a<0.58 else "#f9a825" if a<0.62 else "#00c853" for a in temp["accuracy"]]),
                text=temp.apply(lambda r:f"{r.accuracy:.0%} ({r.n_matches})",axis=1),textposition="outside",textfont=dict(size=9)))
            fig_t.add_hline(y=62.1,line_dash="dash",line_color="#00c853",annotation_text="Overall 62.1%",annotation_font_color="#00c853")
            fig_t.update_layout(height=300,yaxis=dict(title="Accuracy (%)",range=[40,78],gridcolor="#1c2a3e"),**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_t,use_container_width=True)
            st.caption("2020 dip = COVID bubble matches with unusual squad conditions")
        with c2:
            st.markdown('<div class="sh">Accuracy by Confederation</div>',unsafe_allow_html=True)
            cs=conf_acc.sort_values("accuracy",ascending=False)
            fig_conf=go.Figure(go.Bar(x=cs["confederation"],y=cs["accuracy"]*100,
                marker_color=[CONF_COLORS.get(c,"#445566") for c in cs["confederation"]],
                text=cs.apply(lambda r:f"{r.accuracy:.0%} ({r.n_matches:,})",axis=1),textposition="outside",textfont=dict(size=9)))
            fig_conf.add_hline(y=62.1,line_dash="dash",line_color="#00c853")
            fig_conf.update_layout(height=300,yaxis=dict(title="Accuracy (%)",range=[40,78],gridcolor="#1c2a3e"),**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_conf,use_container_width=True)
            st.caption("CONCACAF: clearer strength gaps · CONMEBOL: highly competitive with similar-strength teams")

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
elif page=="🔧  Feature Engineering":
    st.markdown('<div class="wc-banner"><span style="font-size:2rem">🔧</span><div><div class="wc-title">Feature Engineering</div><div class="wc-sub">30 features · 7 groups · 5 data sources · XGBoost (Tuned)</div></div></div>',unsafe_allow_html=True)

    _,_,_,fi,_,_=load_eval()
    ta,tb,tc=st.tabs(["📊 Importance","🗂️ Feature Groups","📈 Distributions"])

    with ta:
        c1,c2=st.columns([1.2,1])
        with c1:
            st.markdown('<div class="sh">All 30 Features — XGBoost Gain</div>',unsafe_allow_html=True)
            n=len(fi)
            bc=["#ffd700" if i==0 else "#00c853" if i<3 else "#1565c0" if i<10 else "#2a3a4a" for i in range(n)]
            fig_a=go.Figure(go.Bar(x=fi["importance"],y=fi["display_name"],orientation="h",
                marker_color=bc,text=fi["importance"].apply(lambda x:f"{x:.4f}"),textposition="outside"))
            fig_a.update_layout(height=680,yaxis=dict(autorange="reversed",tickfont=dict(size=10.5)),
                                xaxis=dict(title="Importance",gridcolor="#1c2a3e"),**PLOTLY_LAYOUT,margin=dict(l=10,r=80,t=10,b=10))
            st.plotly_chart(fig_a,use_container_width=True)
        with c2:
            st.markdown('<div class="sh">Feature Group Treemap</div>',unsafe_allow_html=True)
            gmap={"form_win_pct":"Rolling Form","form_goals_scored":"Rolling Form","form_goals_conceded":"Rolling Form",
                  "form_goal_diff":"Rolling Form","form_clean_sheet_pct":"Rolling Form",
                  "fifa_rank":"FIFA Ranking","fifa_points":"FIFA Ranking","elo_rating":"Elo Rating",
                  "wc_appearances":"WC History","wc_titles":"WC History","wc_best_finish":"WC History",
                  "rank_diff":"Derived","points_diff":"Derived","elo_diff":"Derived",
                  "same_conf":"Match Context","neutral":"Match Context",
                  "match_importance":"Match Importance","h2h_win_rate":"Head-to-Head","h2h_goal_diff":"Head-to-Head"}
            gcol={"Rolling Form":"#1565c0","FIFA Ranking":"#00838f","Elo Rating":"#6a1b9a",
                  "WC History":"#e65100","Derived":"#558b2f","Match Context":"#4e342e",
                  "Match Importance":"#00c853","Head-to-Head":"#f9a825"}
            rows=[]
            for _,r in fi.iterrows():
                b=r["feature"].replace("home_","").replace("away_","")
                rows.append({"feature":r["display_name"],"group":gmap.get(b,"Other"),"importance":r["importance"]})
            fi_g=pd.DataFrame(rows)
            fig_tree=px.treemap(fi_g,path=["group","feature"],values="importance",color="group",color_discrete_map=gcol)
            fig_tree.update_traces(textfont=dict(size=11))
            fig_tree.update_layout(height=340,**{k:v for k,v in PLOTLY_LAYOUT.items()},margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_tree,use_container_width=True)

            st.markdown('<div class="sh">Group Totals</div>',unsafe_allow_html=True)
            gt=fi_g.groupby("group")["importance"].sum().sort_values(ascending=False)
            ti=gt.sum()
            for g,v in gt.items():
                pct=v/ti*100; c=gcol.get(g,"#445566"); bw=int(pct/gt.max()*ti*100/gt.max())
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #1a2030"><div style="width:8px;height:8px;background:{c};border-radius:2px;flex-shrink:0"></div><span style="color:#b0bfcc;font-size:.82rem;flex:1">{g}</span><span style="color:#5a6a7a;font-size:.78rem">{pct:.1f}%</span><div style="width:70px;background:#1c2a3e;border-radius:3px;height:5px"><div style="background:{c};width:{pct:.0f}%;height:100%;border-radius:3px"></div></div></div>',unsafe_allow_html=True)

    with tb:
        groups=[
            ("⚽ Rolling Form","Rolling Form",10,False,"#1565c0","Win rate, goals scored/conceded, goal diff, clean sheet rate — home and away separately. Last 10 matches, shift-1 prevents leakage.","Historical match results (1872–2026)"),
            ("📊 FIFA Ranking","FIFA Ranking",4,False,"#00838f","FIFA rank, points, confederation. Plus rank_diff and points_diff. Limitation: 2026 snapshot used for all historical matches.","api.fifa.com · 211 teams"),
            ("⚡ Elo Rating","Elo Rating",2,False,"#6a1b9a","Long-run Elo strength rating. V2 expanded map (48→108 teams) halved null rate from 70% to 36%.","eloratings.net · current snapshot"),
            ("🏆 WC History","WC History",6,False,"#e65100","Appearances, titles, best finish for both teams. Captures tournament pedigree that form and rankings miss.","Fjelstul WC Database (1930–2022)"),
            ("🎯 Match Importance ⭐","Match Importance",1,True,"#00c853","Score 1–5 by tournament: WC=5, Continental=4, Qualifier=3, Competitive=2, Friendly=1. NEW in V2.","Derived from tournament name"),
            ("🤝 Head-to-Head ⭐","Head-to-Head",2,True,"#f9a825","Win rate and goal diff from last 10 meetings between the two specific teams. NEW in V2.","Historical match results"),
            ("🌍 Match Context","Match Context",2,False,"#4e342e","neutral venue (bool) and same_conf (bool — same FIFA confederation).","Derived"),
        ]
        for title,grp,nf,new,color,desc,src in groups:
            avg=fi_g[fi_g["group"]==grp]["importance"].sum()/fi_g["importance"].sum()*100 if grp in fi_g["group"].values else 0
            badge='<span style="background:#00c853;color:#000;font-size:.65rem;padding:2px 6px;border-radius:4px;font-weight:700">NEW V2</span>' if new else ""
            st.markdown(f'<div class="cc" style="border-left:5px solid {color}"><div style="display:flex;justify-content:space-between;align-items:center">'
                       f'<span style="font-size:.98rem;font-weight:700">{title}</span>'
                       f'<span>{badge} <span style="color:#5a6a7a;font-size:.78rem">{nf} feature{"s" if nf>1 else ""} · {avg:.1f}% of model</span></span></div>'
                       f'<div style="color:#7a8fa0;font-size:.82rem;margin-top:8px;line-height:1.5">{desc}</div>'
                       f'<div style="color:#3a4a5a;font-size:.73rem;margin-top:5px">📦 {src}</div></div>',unsafe_allow_html=True)

    with tc:
        features=load_features()
        cf=features[features["match_importance"]>1]
        feat_choice=st.selectbox("Select feature",
            ["elo_diff","rank_diff","points_diff","home_form_win_pct","match_importance","h2h_win_rate"],
            format_func=lambda x:x.replace("_"," ").title())
        c1,c2=st.columns(2)
        with c1:
            fig_h=px.histogram(cf.dropna(subset=[feat_choice]),x=feat_choice,color="result",
                color_discrete_map={"home_win":"#00c853","draw":"#f9a825","away_win":"#ff5252"},
                nbins=40,barmode="overlay",opacity=.7,labels={feat_choice:feat_choice.replace("_"," ").title()})
            fig_h.update_layout(height=300,legend=dict(orientation="h",y=1.1),**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_h,use_container_width=True)
        with c2:
            fig_b=px.box(cf.dropna(subset=[feat_choice]),x="result",y=feat_choice,color="result",
                color_discrete_map={"home_win":"#00c853","draw":"#f9a825","away_win":"#ff5252"})
            fig_b.update_layout(height=300,showlegend=False,**PLOTLY_LAYOUT,margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_b,use_container_width=True)

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)


# ===========================================================================
# ABOUT  (includes pipeline summary)
# ===========================================================================
elif page=="ℹ️  About":
    st.markdown('<div class="wc-banner"><span style="font-size:2rem">ℹ️</span><div><div class="wc-title">About This Project</div><div class="wc-sub">FIFA World Cup 2026 Prediction · End-to-end ML Pipeline · Portfolio Project</div></div></div>',unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("### Methodology")
        st.markdown('<div class="cc" style="font-size:.87rem;line-height:1.8;color:#b0bfcc"><strong style="color:#fff">Algorithm:</strong> XGBoost (Tuned) via RandomizedSearchCV<br><strong style="color:#fff">Target:</strong> home_win · draw · away_win<br><strong style="color:#fff">Train:</strong> 16,294 competitive matches (1916–2017)<br><strong style="color:#fff">Test:</strong> 5,925 competitive matches (2018–2026)<br><strong style="color:#fff">Split:</strong> Temporal — strict no-leakage policy<br><strong style="color:#fff">CV:</strong> TimeSeriesSplit (5 folds)<br><strong style="color:#fff">Primary metric:</strong> Log Loss (probability calibration)<br><strong style="color:#fff">Simulation:</strong> 10,000 Monte Carlo runs · pre-computed pairwise win probabilities · draws split 50/50 in knockout rounds</div>',unsafe_allow_html=True)

        st.markdown("### Data Sources")
        for icon,name,source,note in [
            ("📊","Historical Match Results","Kaggle","49,490 matches · 1872–2026"),
            ("🏆","FIFA World Rankings","api.fifa.com","211 teams · current snapshot"),
            ("⚡","Elo Ratings","eloratings.net","244 teams · current snapshot"),
            ("🌍","World Cup History","Fjelstul Database","1930–2022"),
            ("📅","2026 WC Fixtures","inside.fifa.com","90 matches"),
        ]:
            st.markdown(f'<div style="display:flex;gap:12px;padding:7px 0;border-bottom:1px solid #1a2030;font-size:.84rem"><span>{icon}</span><div><strong style="color:#fff">{name}</strong><br><span style="color:#5a6a7a">{source} · {note}</span></div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.link_button("⭐ View Full Repository on GitHub","https://github.com/deepalibhattarai92-debug/fifa-world-cup-prediction")

    with c2:
        st.markdown("### Limitations")
        for icon,title,detail,color in [
            ("⚠️","FIFA rankings: 2026 snapshot only","All historical matches use today's rankings. Historical ranking time series would significantly improve this.","#f9a825"),
            ("⚠️","Elo: 36% null in training","~136 minor/defunct nations imputed to median. Model can't distinguish them from truly median-strength teams.","#f9a825"),
            ("❌","No player-level data","No injuries, suspensions or squad depth. Team treated as static.","#ff5252"),
            ("⚠️","62.1% accuracy ceiling","Commercial models with paid data sit at 65–68%. Remaining gap requires player, injury, and tactical data.","#f9a825"),
        ]:
            st.markdown(f'<div style="background:#0f1628;border-radius:8px;padding:11px;border-left:3px solid {color};margin-bottom:8px"><span style="font-size:.88rem">{icon} <strong style="color:#fff">{title}</strong></span><div style="color:#5a6a7a;font-size:.78rem;margin-top:3px">{detail}</div></div>',unsafe_allow_html=True)

        st.markdown("### Version History")
        for ver,acc,ll,note,current in [
            ("V1 — Baseline","59.2%","0.886","All matches · default XGBoost · 27 features",False),
            ("V2 — Final (XGBoost Tuned)","62.1%","0.851","Competitive only · tuned · 30 features",True),
        ]:
            border="border:1.5px solid #00c853;" if current else "border:1px solid #1c3060;"
            label='<span style="background:#00c853;color:#000;font-size:.65rem;padding:1px 5px;border-radius:3px;font-weight:700;margin-left:6px">CURRENT</span>' if current else ""
            st.markdown(f'<div style="background:#0f1628;border-radius:8px;padding:10px 14px;{border}margin-bottom:8px;font-size:.84rem"><strong style="color:#{"00c853" if current else "fff"}">{ver}</strong>{label}<br><span style="color:#7a8fa0">Accuracy: <strong style="color:#fff">{acc}</strong> · Log Loss: <strong style="color:#fff">{ll}</strong></span><br><span style="color:#3a4a5a">{note}</span></div>',unsafe_allow_html=True)

    st.divider()
    st.markdown("### Pipeline Overview")
    steps=[
        ("📥","Data Collection","5 scripts · 49,490 matches · FIFA rankings · Elo · WC history · 2026 fixtures","#00c853"),
        ("🧹","Preprocessing","Name standardisation · former-name alias table · 108-team Elo code map","#1565c0"),
        ("⚙️","Feature Engineering","30 features · rolling form · FIFA · Elo · WC history · match importance · H2H","#00838f"),
        ("🤖","Model Training","XGBoost Tuned · RandomizedSearchCV 40×5-fold TimeSeriesSplit · competitive matches only","#6a1b9a"),
        ("🎲","Simulation","10,000 Monte Carlo runs · 182 pre-computed pairwise win probabilities","#e65100"),
        ("📊","Dashboard","Streamlit · Plotly · 6 pages · all eval artifacts pre-computed","#f9a825"),
    ]
    pipe_cols=st.columns(len(steps))
    for col,(icon,title,desc,color) in zip(pipe_cols,steps):
        col.markdown(f'<div class="ps" style="border-left-color:{color};padding:12px 14px"><div style="font-size:1.2rem">{icon}</div><div style="font-size:.82rem;font-weight:700;color:#fff;margin-top:4px">{title}</div><div style="font-size:.73rem;color:#5a6a7a;margin-top:4px;line-height:1.4">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="ft">⚡ Human curiosity. AI execution. Built by Deepali using Cursor and ChatGPT as engineering partners.</div>',unsafe_allow_html=True)
