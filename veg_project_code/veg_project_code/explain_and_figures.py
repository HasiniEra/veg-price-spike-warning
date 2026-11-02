import warnings, joblib, json, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
M = joblib.load("fitted_models.joblib"); tr, va, te, full = pd.read_pickle("splits.pkl"); P = pd.read_pickle("test_predictions.pkl")
T = "spike2"; y = te[T].values; FS = M["FS"]; IT = M["IT"]
qda, qcols, qparams = M["fitted"]["QDA"]; lda, lcols, lparams = M["fitted"]["LDA (new)"]
pq = P["QDA"].values; plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})
out = {}
show = ["QDA", "LDA (new)", "Logistic Reg.", "Random Forest", "XGBoost", "LightGBM", "EBM (benchmark)", "Gaussian Process (new, 2nd)"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for n in show:
    f, t_, _ = roc_curve(y, P[n]); ax[0].plot(f, t_, label=f"{n} ({roc_auc_score(y,P[n]):.2f})", lw=2 if n == "QDA" else 1)
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("False-alarm rate"); ax[0].set_ylabel("Spikes caught"); ax[0].set_title("ROC curves (test, AUC)"); ax[0].legend(fontsize=6.5)
rates = np.linspace(0.02, 0.4, 20)
for n in ["QDA", "LDA (new)", "Random Forest", "LightGBM", "Gaussian Process (new, 2nd)"]:
    pr = [y[np.argsort(-P[n].values)[:int(r * len(y))]].mean() for r in rates]; ax[1].plot(100 * rates, 100 * np.array(pr), label=n, lw=2 if n == "QDA" else 1)
ax[1].axhline(100 * y.mean(), color="k", ls="--", lw=.8, label="Chance (18.8%)"); ax[1].set_xlabel("% of vegetable-weeks flagged"); ax[1].set_ylabel("Alerts that were real spikes (%)"); ax[1].set_title("Precision vs alert rate"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/fig1_roc_precision.png", dpi=170); plt.close()
pv_model = qda.__class__ if False else None
mq = M["fitted"]["QDA"]; from sklearn.pipeline import make_pipeline
import sklearn.base as skb
val_m = skb.clone(mq[0]).fit(tr[qcols], tr[T]); pv = val_m.predict_proba(va[qcols])[:, 1]
ths = np.linspace(0.05, 0.9, 86); th_f1 = ths[int(np.argmax([f1_score(va[T], pv >= t) for t in ths]))]
rows = []
def rep(lbl, a):
    tn, fp, fn, tp = confusion_matrix(y, a).ravel(); rows.append(dict(rule=lbl, alerts=int(a.sum()), accuracy=100*accuracy_score(y, a), balanced_acc=100*balanced_accuracy_score(y, a), precision=100*precision_score(y, a, zero_division=0), recall=100*recall_score(y, a), F1=100*f1_score(y, a), TP=tp, FP=fp, FN=fn, TN=tn))
rep("Never alert (baseline)", np.zeros_like(y)); rep("Prob >= 0.5", pq >= 0.5); rep(f"Prob >= {th_f1:.2f} (best F1 on validation)", pq >= th_f1)
for f in (0.05, 0.10, 0.20): a = np.zeros_like(y); a[np.argsort(-pq)[:int(f * len(pq))]] = 1; rep(f"Top {int(f*100)}% riskiest weeks", a)
D = pd.DataFrame(rows).round(1); D.to_csv("decision_table_qda.csv", index=False); print("DECISION TABLE (QDA, test):\n", D.to_string(index=False))
bins = pd.cut(pq, [0, .1, .2, .3, .4, .5, .7, 1.0]); cal = pd.DataFrame({"b": bins, "p": pq, "y": y}).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).dropna()
print("\nCALIBRATION (QDA):\n", cal.round(3).to_string()); cal.to_csv("calibration_qda.csv")
fig, ax = plt.subplots(figsize=(4.6, 4)); ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.plot(cal.pred, cal.actual, "o-"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Actual spike rate"); ax.set_title("Calibration (QDA, test)")
for _, r in cal.iterrows(): ax.annotate(int(r.n), (r.pred, r.actual), fontsize=6, xytext=(3, -8), textcoords="offset points")
plt.tight_layout(); plt.savefig("figs/fig2_calibration.png", dpi=170); plt.close()
base = roc_auc_score(y, pq); Xte = te[qcols].copy(); imp = {}
for c in FS:
    dl = []
    for r in range(6): Z = Xte.copy(); Z[c] = np.random.RandomState(r).permutation(Z[c].values); dl.append(base - roc_auc_score(y, qda.predict_proba(Z)[:, 1]))
    imp[c] = float(np.mean(dl))
groups = {"Price level vs normal (26/52-wk median, drawdown, rebound)": ["ratio_med26", "ratio_med52", "drawdown8", "rebound8"], "Recent price moves & volatility": ["chg1", "chg2", "chg4", "vol8", "ratio_ma4"],
          "Market-wide moves": ["market_chg1", "mkt_chg4", "mkt_breadth", "mkt_ratio_med26"], "Season (sin/cos)": ["sin", "cos"], "Dry-zone weather": [c for c in FS if c.startswith("dz_")], "Up-country weather": [c for c in FS if c.startswith("up_")], "Which vegetable": IT}
gimp = {}
for g, cols in groups.items():
    dl = []
    for r in range(6):
        Z = Xte.copy(); idx = np.random.RandomState(r).permutation(len(Z))
        for c in cols: Z[c] = Z[c].values[idx]
        dl.append(base - roc_auc_score(y, qda.predict_proba(Z)[:, 1]))
    gimp[g] = float(np.mean(dl))
s = pd.Series(imp).sort_values(ascending=False); gs_ = pd.Series(gimp).sort_values(ascending=False)
print("\nPERMUTATION IMPORTANCE top 10 (AUC drop):\n", s.head(10).round(4).to_string()); print("\nGROUP importance (AUC drop):\n", gs_.round(4).to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); gs_.sort_values().plot.barh(ax=ax[0], color="#1f77b4"); ax[0].set_title("Importance by feature group"); ax[0].set_xlabel("AUC drop when shuffled")
s.head(12).sort_values().plot.barh(ax=ax[1], color="#d62728"); ax[1].set_title("Top 12 individual features"); ax[1].set_xlabel("AUC drop when shuffled"); plt.tight_layout(); plt.savefig("figs/fig3_importance.png", dpi=170); plt.close()
json.dump(dict(feature=imp, group=gimp), open("importance_qda.json", "w"), indent=1)
pd_feats = [("ratio_med26", "Price vs its 26-week median"), ("drawdown8", "Drop from 8-week high"), ("vol8", "Recent volatility"), ("dz_rain_l1", "Dry-zone rain last week (mm)")]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2)); pdp_out = {}
for a, (f, lab) in zip(ax, pd_feats):
    q = np.unique(np.quantile(te[f], np.linspace(0.03, 0.97, 25))); vals = []
    for v in q: Z = Xte.copy(); Z[f] = v; vals.append(qda.predict_proba(Z)[:, 1].mean())
    a.plot(q, vals, color="#d62728"); a.set_xlabel(lab); a.set_ylabel("Avg predicted spike prob."); a.set_ylim(0, max(0.5, max(vals) * 1.1)); pdp_out[f] = (float(vals[0]), float(vals[-1]))
plt.tight_layout(); plt.savefig("figs/fig4_pdp.png", dpi=170); plt.close(); print("\nPDP (avg prob at lowest vs highest value):", {k: tuple(round(x, 3) for x in v) for k, v in pdp_out.items()})
bg = shap.sample(full[qcols], 40, random_state=0); samp = te[qcols].sample(120, random_state=0)
ex = shap.KernelExplainer(lambda Z: qda.predict_proba(pd.DataFrame(Z, columns=qcols))[:, 1], bg); sv = ex.shap_values(samp, nsamples=150, silent=True)
ma = pd.Series(np.abs(sv).mean(0), index=qcols).sort_values(ascending=False); print("\nSHAP mean|value| top 10 (QDA):\n", ma.head(10).round(4).to_string())
plt.figure(figsize=(6.2, 4)); ma.head(12).sort_values().plot.barh(color="#2ca02c"); plt.title("SHAP: mean |impact| on spike probability (QDA)"); plt.xlabel("mean |SHAP|"); plt.tight_layout(); plt.savefig("figs/fig5_shap.png", dpi=170); plt.close()
te2 = te.copy(); te2["p"] = pq
q1 = pd.qcut(te2.ratio_med26, 5, labels=["much below\nnormal", "below", "normal", "above", "much above"]); r1 = te2.groupby(q1)[T].mean()
q2 = pd.qcut(te2.dz_rain_l1, 5, labels=["driest", "dry", "mid", "wet", "wettest"]); r2 = te2.groupby(q2)[T].mean()
print("\nSpike rate by price vs 26-wk median:", r1.round(3).to_dict()); print("Spike rate by last-week dry-zone rain:", r2.round(3).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); (100 * r1).plot.bar(ax=ax[0], color="#1f77b4", rot=0); ax[0].set_ylabel("Spike within 2 weeks (%)"); ax[0].set_title("By price level vs normal"); ax[0].axhline(100 * y.mean(), color="k", ls="--", lw=.8)
(100 * r2).plot.bar(ax=ax[1], color="#2ca02c", rot=0); ax[1].set_title("By last week's dry-zone rainfall"); ax[1].axhline(100 * y.mean(), color="k", ls="--", lw=.8); plt.tight_layout(); plt.savefig("figs/fig6_domain.png", dpi=170); plt.close()
te2["year"] = te2.week.dt.year; print("\nAUC by test year:"); yr = {}
for yv, g in te2.groupby("year"):
    if g[T].nunique() == 2: yr[int(yv)] = (len(g), float(g[T].mean()), float(roc_auc_score(g[T], g.p)))
print({k: (v[0], round(v[1], 3), round(v[2], 3)) for k, v in yr.items()})
rows = []
for it, g in te2.groupby("Item"):
    if g[T].sum() >= 5 and g[T].nunique() == 2: k = max(1, int(0.2 * len(g))); top = g.sort_values("p", ascending=False).head(k); rows.append(dict(vegetable=it, weeks=len(g), spike_rate=g[T].mean(), AUC=roc_auc_score(g[T], g.p), prec_top20=top[T].mean()))
V = pd.DataFrame(rows).sort_values("AUC"); V.to_csv("by_vegetable_qda.csv", index=False); print("\nBY VEGETABLE (QDA):\n", V.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6.5, 4)); ax.barh(V.vegetable, V.AUC, color=["#d62728" if a < 0.6 else "#1f77b4" for a in V.AUC]); ax.axvline(0.5, color="k", ls="--", lw=.8); ax.set_xlabel("AUC within vegetable (test)"); ax.set_title("Performance varies by vegetable"); plt.tight_layout(); plt.savefig("figs/fig7_by_vegetable.png", dpi=170); plt.close()
json.dump(dict(by_year=yr, th_f1=float(th_f1)), open("robustness.json", "w"))
