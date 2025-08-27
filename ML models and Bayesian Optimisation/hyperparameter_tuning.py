#!/usr/bin/env python3


import pandas as pd
import numpy as np
from math import sqrt

from sklearn.model_selection   import (
    train_test_split, RandomizedSearchCV, GridSearchCV,
    cross_val_score, KFold
)
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import (
    LassoCV, Lasso, RidgeCV, ElasticNetCV
)
from sklearn.svm               import SVR
from sklearn.ensemble          import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics           import r2_score, mean_squared_error
from xgboost                   import XGBRegressor
from sklearn.pipeline          import Pipeline


FILE = "Individual Project Dataset-1.xlsx"
df_data  = pd.read_excel(FILE, sheet_name="My data", engine="openpyxl")
df_feats = pd.read_excel(FILE, sheet_name="element descriptors", engine="openpyxl")
df_data.columns  = df_data.columns.str.strip()
df_feats.columns = df_feats.columns.str.strip()


to_drop = [
    c for c in df_feats.columns
    if df_feats[c].isna().all() or df_feats[c].nunique(dropna=False) <= 1
]
if to_drop:
    df_feats = df_feats.drop(columns=to_drop)


mask = df_data["UTS (MPa)"].notna() & df_data["Elongation"].notna()
df_data  = df_data.loc[mask].reset_index(drop=True)
df_feats  = df_feats.loc[mask].reset_index(drop=True)


composition_cols = [
    "Si","Fe","Cu","Be","Ag","Bi","Pb","Zn","Mn","Mg","Sn","Ti","V",
    "Mo","Ni","Ce","Co","Cr","Li","Sc","Sr","Zr","Al"
]
process_cols = [
    "Extrusion ratio","Extrusion speed (mm/s)","Extrusion Temp",
    "Solution_Temp (℃)","Solution_Time (h)","Quench_Temp (℃)",
    "Aging_Temp (℃)","Aging_Time (h)"
]
missing = [c for c in composition_cols + process_cols if c not in df_data.columns]
if missing:
    raise KeyError(f"Missing columns in My data: {missing}")


df_data[composition_cols] = df_data[composition_cols].fillna(0.0)
df_data[process_cols]     = df_data[process_cols].fillna(-1.0)


X     = pd.concat([
    df_data[composition_cols].astype(float),
    df_feats.astype(float),
    df_data[process_cols].astype(float)
], axis=1)
y_uts = df_data["UTS (MPa)"].astype(float)


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_uts, test_size=0.15, random_state=42
)


imp     = SimpleImputer(strategy="mean").fit(X_tr)
X_tr_i  = imp.transform(X_tr)
X_te_i  = imp.transform(X_te)


sc_lasso = StandardScaler().fit(X_tr_i)
X_tr_s   = sc_lasso.transform(X_tr_i)
X_te_s   = sc_lasso.transform(X_te_i)


def lasso_select(Xtr, ytr):
    cv    = LassoCV(cv=5, n_alphas=100, max_iter=1000000, random_state=0).fit(Xtr, ytr)
    alpha = cv.alpha_
    mdl   = Lasso(alpha=alpha, max_iter=1000000).fit(Xtr, ytr)
    coefs = pd.Series(mdl.coef_, index=X.columns)
    sel   = coefs[coefs.abs() > 1e-8].index.tolist()
    print(f"LASSO: α={alpha:.4f} → {len(sel)} features")
    return sel

sel_lasso = lasso_select(X_tr_s, y_tr)


def gini_select(Xtr, ytr, topn):
    rf   = RandomForestRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    xgbm = XGBRegressor(n_estimators=100, random_state=0, verbosity=0).fit(Xtr, ytr)
    gbdt = GradientBoostingRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr)
    imps = pd.DataFrame({
        "RF":   rf.feature_importances_,
        "XGB":  xgbm.feature_importances_,
        "GBDT": gbdt.feature_importances_
    }, index=X.columns)
    imps["mean_imp"] = imps.mean(axis=1)
    sel = imps["mean_imp"].nlargest(topn).index.tolist()
    print(f"GINI: top {topn} → {len(sel)} features")
    return sel

sel_gini = gini_select(X_tr_i, y_tr, 26)


X_tr_imp = pd.DataFrame(imp.transform(X_tr), columns=X.columns, index=X_tr.index)
X_te_imp = pd.DataFrame(imp.transform(X_te), columns=X.columns, index=X_te.index)
final_feats = sel_gini

# Hyperparameter Tuning

# XGBoost RandomizedSearchCV
xgb_base   = XGBRegressor(objective='reg:squarederror', random_state=42)
param_dist = {
    'n_estimators': [100,200,500],
    'max_depth':    [3,6,10,None],
    'learning_rate':[0.01,0.1,0.3],
    'subsample':    [0.6,0.8,1.0]
}
rs_xgb = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=20, cv=5,
    scoring='r2',
    n_jobs=-1, random_state=42
)

# SVR in a scaling pipeline + GridSearchCV
svr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel='rbf'))
])
svr_param = {
    "svr__C":       [0.1,1,10,100],
    "svr__gamma":   ['scale','auto',0.01,0.1,1],
    "svr__epsilon": [0.01,0.1,0.5]
}
gs_svr = GridSearchCV(
    svr_pipe,
    param_grid=svr_param,
    cv=5, scoring='r2', n_jobs=-1
)

# RidgeCV & ElasticNetCV pipelines (scale + CV)
pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=np.logspace(-3,3,13), cv=5))
])
pipe_enet = Pipeline([
    ("scaler", StandardScaler()),
    ("enet",   ElasticNetCV(
                  alphas=np.logspace(-3,1,10),
                  l1_ratio=[0.2,0.5,0.8],
                  cv=5,
                  max_iter=1000000))
])

# collecting all nested‐CV estimators
nested_estimators = {
    "XGBoost": rs_xgb,
    "SVR":     gs_svr,
    "Ridge":   pipe_ridge,
    "ElasticNet": pipe_enet
}


# Nested 5-fold CV


print("\n=== NESTED 5-FOLD CV (outer) on Training Set ===")
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, est in nested_estimators.items():
    scores = cross_val_score(
        est,
        X_tr_imp[final_feats],
        y_tr,
        cv=outer_cv,
        scoring='r2',
        n_jobs=-1
    )
    print(f"{name:12s}:  R² = {scores.mean():.3f}  ± {scores.std():.3f}")


# fitting each tuned estimator on the full 85% train then eval on hold-out
print("\n=== FINAL UTS TEST PERFORMANCE (hold-out 15%) ===")

# fitting each search on full train
best_xgb   = nested_estimators["XGBoost"].fit(
    X_tr_imp[final_feats], y_tr
).best_estimator_
best_svr   = nested_estimators["SVR"].fit(
    X_tr_imp[final_feats], y_tr
).best_estimator_
best_ridge = nested_estimators["Ridge"].fit(
    X_tr_imp[final_feats], y_tr
).named_steps["ridge"]
best_enet  = nested_estimators["ElasticNet"].fit(
    X_tr_imp[final_feats], y_tr
).named_steps["enet"]

def evaluate(name, model, Xtr, ytr, Xte, yte, scale=False):
    if scale:
        scl = StandardScaler().fit(Xtr)
        Xtr, Xte = scl.transform(Xtr), scl.transform(Xte)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    rmse = sqrt(mean_squared_error(yte,yp))
    print(f"{name:12s} → R² = {r2_score(yte, yp):.3f},  RMSE = {rmse:.3f}")

Xtr_raw = X_tr_imp[final_feats]
Xte_raw = X_te_imp[final_feats]

evaluate("XGBoost",      best_xgb,    Xtr_raw, y_tr, Xte_raw, y_te)
evaluate("SVR",          best_svr,    Xtr_raw, y_tr, Xte_raw, y_te)        # pipeline handles scaling
evaluate("RidgeCV",      best_ridge,  Xtr_raw, y_tr, Xte_raw, y_te, scale=True)
evaluate("ElasticNetCV", best_enet,   Xtr_raw, y_tr, Xte_raw, y_te, scale=True)

import matplotlib.pyplot as plt

def plot_pred_vs_actual(model, name, Xtr, ytr, Xte, yte, save_path = None):
    """
    Scatter‐plot of model.predict vs actual y,
    with ±15% error bands, blue=train, red=test.
    """
    # fitting on train
    model.fit(Xtr, ytr)

    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)


    all_actual = np.concatenate([ytr, yte])
    lo, hi = all_actual.min(), all_actual.max()
    pad = (hi - lo) * 0.05
    xlims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6,6))
    # scatter
    ax.scatter(ytr, ytr_pred, label="train set", s=20)
    ax.scatter(yte, yte_pred, label="test set",  s=20, color='C1')
    # identity line
    ax.plot(xlims, xlims, color='k', linestyle='-')
    # ±15% bands
    ax.plot(xlims, 0.85*np.array(xlims), linestyle='--', color='gray')
    ax.plot(xlims, 1.15*np.array(xlims), linestyle='--', color='gray')

    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xlabel(f"Actual UTS (MPa)", fontsize=10)
    ax.set_ylabel(f"Predicted UTS (MPa)", fontsize=10)
    ax.set_title(f"{name}: Predicted vs Actual UTS", fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi = 300)
    plt.show()


plot_pred_vs_actual(best_xgb,      "XGBoost",      Xtr_raw, y_tr, Xte_raw, y_te, save_path="XGBoost_pred_vs_actual.png")
plot_pred_vs_actual(best_svr,      "SVR",          Xtr_raw, y_tr, Xte_raw, y_te, save_path="SVR_pred_vs_actual.png")
plot_pred_vs_actual(best_ridge,    "RidgeCV",      Xtr_raw, y_tr, Xte_raw, y_te, save_path="RidgeCV_pred_vs_actual.png")
plot_pred_vs_actual(best_enet,     "ElasticNetCV", Xtr_raw, y_tr, Xte_raw, y_te, save_path="ElasticNetCV_pred_vs_actual.png")



