import itertools, time, warnings, joblib, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from veg_features import PRICE, EXTRA, weather_cols
X = pd.read_pickle("features.pkl"); WE = weather_cols(X); FS = PRICE + EXTRA + WE; T = "spike2"
d = X.dropna(subset=FS + [T]).copy().reset_index(drop=True)
tr, va, te = d[d.week < "2023-01-01"].copy(), d[(d.week >= "2023-01-01") & (d.week < "2024-01-01")].copy(), d[d.week >= "2024-01-01"].copy()
full = pd.concat([tr, va]).copy()
lo, hi = tr[FS].quantile(0.01), tr[FS].quantile(0.99)
for df_ in (tr, va, te, full): df_[FS] = df_[FS].clip(lo, hi, axis=1)
items = sorted(d.Item.unique()); IT = ["it_" + i for i in items]
for df_ in (tr, va, te, full):
    for i in items: df_["it_" + i] = (df_.Item == i).astype(float)
    df_["item_code"] = df_.Item.map({i: k for k, i in enumerate(items)}); df_["Item_str"] = df_.Item.astype(str)
print(f"train {len(tr)} | validation {len(va)} | test {len(te)} | spike rate train {tr[T].mean():.3f} val {va[T].mean():.3f} test {te[T].mean():.3f}", flush=True)
y_te = te[T].values; log = []; preds = {}; fitted = {}

def prec_at(y, p, f):
    k = int(f * len(p)); o = np.argsort(-p)[:k]; return y[o].mean(), y[o].sum() / y.sum()
def evaluate(p, y):
    r = dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p)); a5, _ = prec_at(y, p, .05); a10, r10 = prec_at(y, p, .10); a20, r20 = prec_at(y, p, .20)
    k = int(.10 * len(p)); pred = np.zeros_like(y); pred[np.argsort(-p)[:k]] = 1
    r.update({"prec@5%": a5, "prec@10%": a10, "recall@10%": r10, "prec@20%": a20, "recall@20%": r20, "balanced_acc@10%": balanced_accuracy_score(y, pred)}); return r

def cols_for(name, with_item): return FS + (IT if with_item else [])
zoo = {
 "LDA (new)": (lambda g, c: make_pipeline(SimpleImputer(), StandardScaler(), LDA(solver="lsqr", shrinkage=g["shrinkage"])), dict(shrinkage=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], with_item=[False, True])),
 "QDA": (lambda g, c: make_pipeline(SimpleImputer(), StandardScaler(), QDA(reg_param=g["reg_param"])), dict(reg_param=[0.1, 0.3, 0.5, 0.7, 0.9], with_item=[False, True])),
 "Logistic Reg.": (lambda g, c: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=4000, C=g["C"])), dict(C=[0.003, 0.01, 0.03, 0.1, 0.3, 1], with_item=[False, True])),
 "Random Forest": (lambda g, c: make_pipeline(SimpleImputer(), RandomForestClassifier(300, min_samples_leaf=g["leaf"], max_features=g["mf"], random_state=0, n_jobs=-1)), dict(leaf=[10, 30, 60], mf=[0.3, 0.6], with_item=[True])),
 "XGBoost": (lambda g, c: XGBClassifier(max_depth=g["depth"], learning_rate=g["lr"], n_estimators=g["n"], subsample=0.8, colsample_bytree=0.7, random_state=0, verbosity=0), dict(depth=[2, 3, 5], lr=[0.03, 0.1], n=[150, 300], with_item=[True])),
 "LightGBM": (lambda g, c: LGBMClassifier(num_leaves=g["leaves"], learning_rate=g["lr"], n_estimators=g["n"], min_child_samples=g["mcs"], subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1), dict(leaves=[3, 7], lr=[0.03, 0.1], n=[150, 300], mcs=[20, 60], with_item=[True])),
}
chosen = {}
for name, (mk, grid) in zoo.items():
    best = None
    for vals in itertools.product(*grid.values()):
        g = dict(zip(grid.keys(), vals)); cols = cols_for(name, g["with_item"]) if name != "Random Forest" and name not in ("XGBoost", "LightGBM") else FS + ["item_code"]
        m = mk(g, cols).fit(tr[cols], tr[T]); a = roc_auc_score(va[T], m.predict_proba(va[cols])[:, 1])
        log.append(dict(model=name, params=str({k: v for k, v in g.items()}), val_AUC=a))
        if best is None or a > best[0]: best = (a, g, cols)
    chosen[name] = best; print(f"tuned {name:14s} val AUC {best[0]:.3f} {best[1]}", flush=True)
    m = mk(best[1], best[2]).fit(full[best[2]], full[T]); preds[name] = m.predict_proba(te[best[2]])[:, 1]; fitted[name] = (m, best[2], best[1])
best = None; cols = FS + ["Item_str"]
for lr, inter in itertools.product([0.02, 0.05], [0, 8]):
    m = ExplainableBoostingClassifier(learning_rate=lr, interactions=inter, max_rounds=600, outer_bags=4, inner_bags=0, random_state=0, max_bins=64).fit(tr[cols], tr[T])
    a = roc_auc_score(va[T], m.predict_proba(va[cols])[:, 1]); log.append(dict(model="EBM (benchmark)", params=str(dict(lr=lr, interactions=inter)), val_AUC=a))
    if best is None or a > best[0]: best = (a, dict(lr=lr, inter=inter))
print(f"tuned EBM            val AUC {best[0]:.3f} {best[1]}", flush=True)
m = ExplainableBoostingClassifier(learning_rate=best[1]["lr"], interactions=best[1]["inter"], max_rounds=600, outer_bags=4, inner_bags=0, random_state=0, max_bins=64).fit(full[cols], full[T])
preds["EBM (benchmark)"] = m.predict_proba(te[cols])[:, 1]; fitted["EBM (benchmark)"] = (m, cols, best[1])
iv = tr.groupby("Item").chg1.std(); isr = tr.groupby("Item")[T].mean()
for df_ in (tr, full, te): df_["item_vol"] = df_.Item.map(iv); df_["item_spike_rate"] = df_.Item.map(isr); df_["y_log"] = np.log(df_.next2 / df_.price)
GPC = ["ratio_med26", "ratio_med52", "drawdown8", "rebound8", "chg1", "chg4", "vol8", "market_chg1", "mkt_chg4", "mkt_breadth", "item_vol", "item_spike_rate", "dz_rain_l1", "up_rain_l2", "up_temp_l1"]
for df_ in (tr, full, te): df_["y_log"] = df_["y_log"].replace([np.inf, -np.inf], np.nan)
gfull = full.dropna(subset=["y_log"] + GPC); gte = te.copy(); gte[GPC] = gte[GPC].fillna(gfull[GPC].median()); gte["y_log"] = gte["y_log"].fillna(0.0)
sub = np.random.RandomState(0).choice(len(gfull), 1200, replace=False); sc = StandardScaler().fit(gfull.iloc[sub][GPC]); t0 = time.time()
gp = GaussianProcessRegressor(C(0.05, (1e-3, 1)) * RBF(np.ones(len(GPC)) * 3, (0.3, 100)) + WhiteKernel(0.05, (1e-3, 1)), normalize_y=True, random_state=0).fit(sc.transform(gfull.iloc[sub][GPC]), gfull.y_log.values[sub])
mu, sd = gp.predict(sc.transform(gte[GPC]), return_std=True); preds["Gaussian Process (new, 2nd)"] = 1 - norm.cdf((np.log(1.25) - mu) / sd)
okc = te.y_log.notna().values; cover = np.mean((te.y_log.values[okc] >= mu[okc] - 1.96 * sd[okc]) & (te.y_log.values[okc] <= mu[okc] + 1.96 * sd[okc])); print(f"GP fitted in {time.time()-t0:.0f}s | 95% interval coverage {100*cover:.0f}%", flush=True)
fitted["GP"] = (gp, sc, GPC, dict(item_vol=iv.to_dict(), item_spike_rate=isr.to_dict()))
rows = [dict(model=n, **evaluate(p, y_te)) for n, p in preds.items()]
rows.append(dict(model="Random alerts (chance)", AUC=0.5, AP=y_te.mean(), **{"prec@5%": y_te.mean(), "prec@10%": y_te.mean(), "recall@10%": 0.10, "prec@20%": y_te.mean(), "recall@20%": 0.20, "balanced_acc@10%": 0.5}))
R = pd.DataFrame(rows); print("\nFINAL TEST: 2024 - Mar 2026 | spike within 2 weeks | base rate %.3f | n=%d\n" % (y_te.mean(), len(y_te))); print(R.round(3).to_string(index=False))
rng = np.random.RandomState(0); weeks = te.week.unique(); wk_idx = {w: np.where(te.week.values == w)[0] for w in weeks}; B = []
for _ in range(400):
    idx = np.concatenate([wk_idx[w] for w in rng.choice(weeks, len(weeks))]); y = y_te[idx]
    if y.min() == y.max(): continue
    B.append({n: roc_auc_score(y, p[idx]) for n, p in preds.items()})
B = pd.DataFrame(B); ci = pd.DataFrame({"AUC_lo": B.quantile(.025), "AUC_hi": B.quantile(.975)})
diff = pd.DataFrame({n: [(B["LDA (new)"] - B[n]).mean(), (B["LDA (new)"] - B[n]).quantile(.025), (B["LDA (new)"] - B[n]).quantile(.975)] for n in B.columns if n != "LDA (new)"}, index=["mean_diff", "lo", "hi"]).T
print("\nAUC 95% CI (bootstrap over weeks):"); print(ci.round(3).to_string()); print("\nLDA minus other (AUC difference, 95% CI):"); print(diff.round(3).to_string())
R.merge(ci, left_on="model", right_index=True, how="left").to_csv("test_results.csv", index=False); pd.DataFrame(log).to_csv("tuning_log.csv", index=False); diff.to_csv("lda_vs_others.csv")
te_out = te[["Item", "week", "price", "next2", "spike2", "chg_n2"]].copy()
for n, p in preds.items(): te_out[n] = p
te_out.to_pickle("test_predictions.pkl"); joblib.dump(dict(fitted=fitted, FS=FS, IT=IT, items=items, lo=lo, hi=hi, GPC=GPC), "fitted_models.joblib")
pd.to_pickle((tr, va, te, full), "splits.pkl")
