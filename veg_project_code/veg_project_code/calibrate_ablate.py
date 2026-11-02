import warnings, joblib, json, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import sklearn.base as skb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = joblib.load("fitted_models.joblib"); tr, va, te, full = pd.read_pickle("splits.pkl"); T = "spike2"; y = te[T].values
qda, qcols, qparams = M["fitted"]["QDA"]; FS = M["FS"]; IT = M["IT"]
mA = skb.clone(qda).fit(tr[qcols], tr[T]); pv = mA.predict_proba(va[qcols])[:, 1]; pt = mA.predict_proba(te[qcols])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(pv, va[T].values); pc = iso.predict(pt)
print(f"QDA (train-only): AUC raw {roc_auc_score(y,pt):.3f} | calibrated {roc_auc_score(y,pc):.3f} | Brier raw {brier_score_loss(y,pt):.3f} -> calibrated {brier_score_loss(y,pc):.3f} (no-skill {brier_score_loss(y,np.full(len(y),y.mean())):.3f})")
bins = pd.cut(pc, [0, .1, .2, .3, .4, .5, .7, 1.0]); cal = pd.DataFrame({"b": bins, "p": pc, "y": y}).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).dropna(); print("\nCALIBRATED reliability on TEST:\n", cal.round(3).to_string())
rows = []
for th in (0.3, 0.4, 0.5):
    a = (pc >= th).astype(int); rows.append(dict(rule=f"Calibrated prob >= {th}", alerts=int(a.sum()), accuracy=100*accuracy_score(y, a), balanced_acc=100*balanced_accuracy_score(y, a), precision=100*precision_score(y, a, zero_division=0), recall=100*recall_score(y, a), F1=100*f1_score(y, a)))
print("\n", pd.DataFrame(rows).round(1).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
for a, (lab, p) in zip(ax, [("QDA raw", pt), ("QDA calibrated (isotonic)", pc)]):
    b = pd.cut(p, [0, .1, .2, .3, .4, .5, .7, 1.0]); c = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).dropna()
    a.plot([0, 1], [0, 1], "k--", lw=.8); a.plot(c.pred, c.actual, "o-"); a.set_title(lab); a.set_xlabel("Predicted probability"); a.grid(alpha=.3)
    for _, r in c.iterrows(): a.annotate(int(r.n), (r.pred, r.actual), fontsize=6, xytext=(3, -8), textcoords="offset points")
ax[0].set_ylabel("Actual spike rate (test)"); plt.tight_layout(); plt.savefig("figs/fig2_calibration.png", dpi=170); plt.close()
weather = [c for c in FS if c.startswith(("up_", "dz_"))]; noW = [c for c in qcols if c not in weather]
m_full = skb.clone(qda).fit(full[qcols], full[T]); m_now = skb.clone(qda).fit(full[noW], full[T]); pf, pn = m_full.predict_proba(te[qcols])[:, 1], m_now.predict_proba(te[noW])[:, 1]
rng = np.random.RandomState(0); weeks = te.week.unique(); idxs = {w: np.where(te.week.values == w)[0] for w in weeks}; d = []
for _ in range(400):
    i = np.concatenate([idxs[w] for w in rng.choice(weeks, len(weeks))]); yy = y[i]
    if yy.min() != yy.max(): d.append(roc_auc_score(yy, pf[i]) - roc_auc_score(yy, pn[i]))
d = np.array(d)
def p10(p): k = int(.1 * len(p)); return y[np.argsort(-p)[:k]].mean()
print(f"\nABLATION (QDA): no weather AUC {roc_auc_score(y,pn):.3f}, prec@10% {p10(pn):.3f} | with weather AUC {roc_auc_score(y,pf):.3f}, prec@10% {p10(pf):.3f} | AUC gain {d.mean():+.3f} 95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] helps in {100*(d>0).mean():.0f}% of resamples")
joblib.dump(dict(model=mA, iso=iso, cols=qcols, FS=FS, IT=IT, items=M["items"], lo=M["lo"], hi=M["hi"]), "app_model.joblib")
json.dump(dict(auc_raw=float(roc_auc_score(y, pt)), auc_cal=float(roc_auc_score(y, pc)), brier_raw=float(brier_score_loss(y, pt)), brier_cal=float(brier_score_loss(y, pc)), weather_gain=float(d.mean()), weather_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], auc_no_weather=float(roc_auc_score(y, pn)), auc_with_weather=float(roc_auc_score(y, pf))), open("calibration_ablation.json", "w"), indent=1)
