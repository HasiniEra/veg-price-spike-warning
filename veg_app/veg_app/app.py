from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA, ASSETS = HERE / "data", HERE / "assets"
st.set_page_config(page_title="Vegetable price-spike early warning", page_icon="🥕", layout="wide")

META = ["Item", "week", "price", "next2", "chg_n2", "spike2"]
GROUPS = {
    "Price level vs normal": ["ratio_med26", "ratio_med52", "drawdown8", "rebound8"],
    "Recent price moves & volatility": ["chg1", "chg2", "chg4", "vol8", "ratio_ma4"],
    "Market-wide moves": ["market_chg1", "mkt_chg4", "mkt_breadth", "mkt_ratio_med26"],
    "Season": ["sin", "cos"],
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA / "veg_app_data.csv", parse_dates=["week"])
    return df.sort_values(["Item", "week"]).reset_index(drop=True)


@st.cache_resource
def train():
    df = load_data()
    FS = [c for c in df.columns if c not in META]
    items = sorted(df.Item.unique())
    d = df.dropna(subset=FS + ["spike2"])
    tr, va = d[d.week < "2023-01-01"], d[(d.week >= "2023-01-01") & (d.week < "2024-01-01")]
    lo, hi = tr[FS].quantile(0.01), tr[FS].quantile(0.99)

    def design(x: pd.DataFrame) -> pd.DataFrame:
        z = x[FS].clip(lo, hi, axis=1).copy()
        for i in items:
            z["it_" + i] = (x.Item == i).astype(float)
        return z

    model = make_pipeline(SimpleImputer(), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.5)).fit(design(tr), tr.spike2)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(model.predict_proba(design(va))[:, 1], va.spike2)
    neutral = design(tr).median()
    for i in items:
        neutral["it_" + i] = float((tr.Item == i).mean())
    groups = dict(GROUPS)
    groups["Dry-zone weather"] = [c for c in FS if c.startswith("dz_")]
    groups["Up-country weather"] = [c for c in FS if c.startswith("up_")]
    groups["Which vegetable"] = ["it_" + i for i in items]
    return dict(model=model, iso=iso, design=design, FS=FS, items=items, neutral=neutral, groups=groups, base=float(d[d.week >= "2024-01-01"].spike2.mean()))


def prob(M, rows: pd.DataFrame) -> np.ndarray:
    return M["iso"].predict(M["model"].predict_proba(M["design"](rows))[:, 1])


def prob_matrix(M, X: pd.DataFrame) -> np.ndarray:
    return M["iso"].predict(M["model"].predict_proba(X)[:, 1])


def risk_label(p: float) -> str:
    return "🔴 High" if p >= 0.40 else ("🟠 Medium" if p >= 0.20 else "🟢 Low")


df = load_data()
M = train()
FS, items = M["FS"], M["items"]
ready = df.dropna(subset=FS)

st.sidebar.title("🥕 Choose a vegetable")
item = st.sidebar.selectbox("Vegetable (Dambulla wholesale)", items, index=items.index("Tomato") if "Tomato" in items else 0)
weeks = ready[ready.Item == item].week.sort_values(ascending=False)
week = st.sidebar.selectbox("Week starting", list(weeks), format_func=lambda w: w.strftime("%d %b %Y"))
st.sidebar.caption("Prices: Central Bank of Sri Lanka. Weather: NASA POWER for 7 Sri Lankan growing areas.")

st.title("Vegetable price-spike early warning")
st.caption("Chance that this vegetable's wholesale price rises by more than 25% within the next 2 weeks.")
row = ready[(ready.Item == item) & (ready.week == week)].iloc[[0]]
p = float(prob(M, row)[0])
tab1, tab2, tab3, tab4 = st.tabs(["🚨 Spike risk", "🔧 What-if", "🥬 All vegetables", "📊 Model, validation & fairness"])

with tab1:
    c = st.columns(4)
    c[0].metric("Spike chance (next 2 weeks)", f"{100 * p:.0f}%", f"{100 * (p - M['base']):+.0f} pts vs average week")
    c[1].metric("Risk level", risk_label(p))
    c[2].metric("Price this week (Rs/kg)", f"{row.price.iloc[0]:.0f}")
    c[3].metric("Price vs its 26-week median", f"{100 * row.ratio_med26.iloc[0]:+.0f}%")
    nxt = row.next2.iloc[0]
    if pd.notna(nxt):
        chg = nxt / row.price.iloc[0] - 1
        (st.success if chg > 0.25 else st.info)(
            f"What actually happened: the price 2 weeks later was Rs {nxt:.0f} ({100 * chg:+.0f}%). "
            f"{'This WAS a spike.' if chg > 0.25 else 'No spike.'}")
    else:
        st.info("This is the latest week in the data, so the outcome is not known yet.")

    left, right = st.columns([3, 2])
    with left:
        hist = df[(df.Item == item) & (df.week <= week)].tail(40)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(hist.week, hist.price, "o-", ms=3, color="#1f77b4", label="Wholesale price")
        ax.plot(hist.week, hist.price / (1 + hist.ratio_med26), "--", color="grey", label="26-week median")
        ax.axvline(week, color="#d62728", lw=1); ax.set_ylabel("Rs per kg"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.autofmt_xdate(); st.pyplot(fig, clear_figure=True)
    with right:
        base_x = M["design"](row)
        contrib = {}
        for g, cols in M["groups"].items():
            Z = base_x.copy(); Z[cols] = M["neutral"][cols].values
            contrib[g] = p - float(prob_matrix(M, Z)[0])
        s = pd.Series(contrib).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.barh(s.index, 100 * s.values, color=["#d62728" if v > 0 else "#2ca02c" for v in s.values]); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("Effect on spike chance (percentage points)"); ax.set_title("Why this risk? (red = raises risk)", fontsize=9)
        ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig, clear_figure=True)
    st.caption("Reasons are found by replacing each group of inputs with its typical value and measuring how the probability changes.")

with tab2:
    st.subheader("Change the inputs and see the risk move")
    r0 = row.iloc[0]
    a, b = st.columns(2)
    v_med = a.slider("Price vs 26-week median", -0.7, 1.5, float(np.clip(r0.ratio_med26, -0.7, 1.5)), 0.01, format="%.2f")
    v_dd = b.slider("Drop from 8-week high", -0.9, 0.0, float(np.clip(r0.drawdown8, -0.9, 0.0)), 0.01, format="%.2f")
    a, b = st.columns(2)
    v_vol = a.slider("Recent volatility (std of weekly % change)", 0.0, 0.8, float(np.clip(r0.vol8, 0, 0.8)), 0.01, format="%.2f")
    v_rain = b.slider("Dry-zone rainfall last week (mm, weekly total)", 0.0, float(max(250, r0.dz_rain_l1 * 1.2)), float(r0.dz_rain_l1), 1.0)
    X2 = row.copy(); X2["ratio_med26"], X2["drawdown8"], X2["vol8"], X2["dz_rain_l1"] = v_med, v_dd, v_vol, v_rain
    p2 = float(prob(M, X2)[0])
    c = st.columns(3)
    c[0].metric("Original chance", f"{100 * p:.0f}%"); c[1].metric("What-if chance", f"{100 * p2:.0f}%", f"{100 * (p2 - p):+.0f} pts"); c[2].metric("Risk level", risk_label(p2))
    grid = np.linspace(0, max(250, r0.dz_rain_l1 * 1.2), 40); Xg = pd.concat([X2] * len(grid), ignore_index=True); Xg["dz_rain_l1"] = grid
    pg = prob(M, Xg)
    fig, ax = plt.subplots(figsize=(6.5, 3)); ax.plot(grid, 100 * pg, color="#d62728"); ax.scatter([v_rain], [100 * p2], color="k", zorder=5)
    ax.set_xlabel("Dry-zone rainfall last week (mm)"); ax.set_ylabel("Spike chance (%)"); ax.set_title(f"{item}: risk vs last week's rain (other inputs as above)", fontsize=9); ax.grid(alpha=.3)
    st.pyplot(fig, clear_figure=True)
    st.caption("Changing one input while keeping the others fixed can create combinations that rarely occur in real life, so treat this as a guide to how the model reasons, not a forecast.")

with tab3:
    st.subheader(f"All vegetables, week starting {week.strftime('%d %b %Y')}")
    allw = ready[ready.week == week].copy()
    if allw.empty:
        st.info("No complete data for the other vegetables in this week.")
    else:
        allw["p"] = prob(M, allw)
        t = pd.DataFrame({"Vegetable": allw.Item, "Price (Rs/kg)": allw.price.round(0).astype(int), "Price vs 26-wk median": (100 * allw.ratio_med26).round(0).astype(int).astype(str) + "%",
                          "Spike chance": (100 * allw.p).round(0).astype(int).astype(str) + "%", "Risk": allw.p.map(risk_label), "_p": allw.p}).sort_values("_p", ascending=False).drop(columns="_p")
        st.dataframe(t, hide_index=True)
        st.caption("Sorted by spike chance. Imported potato and onion prices are mostly policy-driven and are poorly predicted; see the fairness tab.")

with tab4:
    st.subheader("How Built and Tested?")
    st.markdown("**Data:** Central Bank of Sri Lanka daily vegetable prices (Aug 2016 - Mar 2026, Dambulla wholesale, 15 vegetables, 5,390 labelled weeks) and NASA POWER daily weather for seven Sri Lankan growing areas.  \n"
                "**Split by time:** train 2016-2022, validate 2023 (tuning and probability calibration), **test once on 2024 to March 2026** (1,503 vegetable-weeks, 18.8% are real spikes).  \n"
                "**New algorithm:** Quadratic Discriminant Analysis (not covered in lectures), compared with Logistic Regression, Random Forest, XGBoost, LightGBM, LDA, EBM and a Gaussian Process.")
    tr_ = pd.read_csv(DATA / "test_results.csv")[["model", "AUC", "AUC_lo", "AUC_hi", "AP", "prec@5%", "prec@10%", "recall@10%", "balanced_acc@10%"]].round(3)
    st.dataframe(tr_, hide_index=True)
    st.markdown("**Alert rules for the app model (test period).** Plain accuracy is about 81% for a model that never alerts, so precision and balanced accuracy matter more.")
    st.dataframe(pd.read_csv(DATA / "decision_table.csv"), hide_index=True)
    for f, cap in [("fig1_roc_precision.png", "ROC curves and precision against the share of weeks flagged."), ("fig2_calibration.png", "Calibration: isotonic calibration makes the probabilities honest."),
                   ("fig3_importance.png", "What the model uses (permutation importance)."), ("fig4_pdp.png", "Partial dependence of spike chance on key inputs."),
                   ("fig6_domain.png", "Spike rates by price level and by last week's rainfall."), ("fig7_by_vegetable.png", "Performance differs by vegetable.")]:
        st.image(str(ASSETS / f)); st.caption(cap)
    st.warning("**Limitations and fairness.** The model is strong for perishable local vegetables (snake gourd, beans, brinjal, lime, green chilli) and weak for imported potato and onions, whose prices are driven by imports and policy. "
               "Weather comes from a coarse satellite product (about 50 km), not rain gauges, and the supply areas are my assumption. Only one market (Dambulla wholesale) is modelled and the test covers about 2 years. "
               "A 25% threshold was chosen. The tool supports decisions; it does not replace market knowledge.")
