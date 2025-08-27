

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

FILE = "Individual Project Dataset-1.xlsx"

df_data = pd.read_excel(FILE, sheet_name="My data", header=0, engine="openpyxl")
df_data.columns = df_data.columns.str.strip()

df_feats = pd.read_excel(FILE, sheet_name="element descriptors", engine="openpyxl")
df_feats.columns = df_feats.columns.str.strip()

to_drop = [c for c in df_feats.columns 
           if df_feats[c].isna().all() or df_feats[c].nunique(dropna=False) <= 1]
if to_drop:
    print("Dropping constant/empty descriptor cols:", to_drop)
    df_feats = df_feats.drop(columns=to_drop)

mask = df_data["UTS (MPa)"].notna() & df_data["Elongation"].notna()
df_data = df_data.loc[mask].reset_index(drop=True)
df_feats = df_feats.loc[mask].reset_index(drop=True)

composition_cols = [
    "Si","Fe","Cu","Be","Ag","Bi","Pb","Zn",
    "Mn","Mg","Sn","Ti","V","Mo","Ni","Ce","Co","Cr",
    "Li","Sc","Sr","Zr","Al"
]
process_cols = [
    "Extrusion ratio",
    "Extrusion speed (mm/s)",
    "Extrusion Temp",
    "Solution_Temp (℃)",
    "Solution_Time (h)",
    "Quench_Temp (℃)",
    "Aging_Temp (℃)",
    "Aging_Time (h)"
]

# Sanity‐check 
missing = [c for c in composition_cols + process_cols if c not in df_data.columns]
if missing:
    raise KeyError(f"Missing columns in My data: {missing}")

df_data[composition_cols] = df_data[composition_cols].fillna(0.0)

# Building feature matrix and targets
df_data[process_cols] = df_data[process_cols].fillna(-1)

X = pd.concat([
    df_data[composition_cols].astype(float),
    df_feats.astype(float),
    df_data[process_cols].astype(float)
], axis=1)
y_uts  = df_data["UTS (MPa)"].astype(float)
y_elon = df_data["Elongation"].astype(float)

# Train/test split 85/15
X_tr_uts, X_te_uts, y_tr_uts, y_te_uts = train_test_split(
    X, y_uts, test_size=0.15, random_state=42
)
X_tr_el, X_te_el, y_tr_el, y_te_el = train_test_split(
    X, y_elon, test_size=0.15, random_state=42
)

# Impute missing values (mean) separately for each split
imp_uts = SimpleImputer(strategy="mean").fit(X_tr_uts)
X_tr_uts_i = imp_uts.transform(X_tr_uts)
X_te_uts_i = imp_uts.transform(X_te_uts)

imp_el = SimpleImputer(strategy="mean").fit(X_tr_el)
X_tr_el_i = imp_el.transform(X_tr_el)
X_te_el_i = imp_el.transform(X_te_el)

# Scaling for Lasso - note that trees don’t need scaling
# check the report for this explanation
scaler_uts = StandardScaler().fit(X_tr_uts_i)
X_tr_uts_s = scaler_uts.transform(X_tr_uts_i)
X_te_uts_s = scaler_uts.transform(X_te_uts_i)

scaler_el = StandardScaler().fit(X_tr_el_i)
X_tr_el_s = scaler_el.transform(X_tr_el_i)
X_te_el_s = scaler_el.transform(X_te_el_i)

def lasso_select(X_tr, y_tr, X_te, y_te, target_name):
    cv = LassoCV(cv=5, n_alphas=100, random_state=0).fit(X_tr, y_tr)
    alpha = cv.alpha_
    model = Lasso(alpha=alpha).fit(X_tr, y_tr)
    coefs = pd.Series(model.coef_, index=X.columns)
    selected = coefs[coefs.abs() > 1e-8].index.tolist()
    print(f"\nLASSO ({target_name}): α={alpha:.4f} → {len(selected)} features")
    print(selected)
    return selected

sel_lasso_uts  = lasso_select(X_tr_uts_s, y_tr_uts, X_te_uts_s, y_te_uts, "UTS")
sel_lasso_elon = lasso_select(X_tr_el_s,  y_tr_el,  X_te_el_s,  y_te_el,  "Elongation")

def gini_select(X_tr, y_tr, n_top, target_name):
    rf   = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    xgbm = xgb.XGBRegressor(n_estimators=100, random_state=0, verbosity=0).fit(X_tr, y_tr)
    gbdt = GradientBoostingRegressor(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    imps = pd.DataFrame({
        "RF":   rf.feature_importances_,
        "XGB":  xgbm.feature_importances_,
        "GBDT": gbdt.feature_importances_
    }, index=X.columns)
    imps["mean_imp"] = imps.mean(axis=1)
    topn = imps["mean_imp"].nlargest(n_top).index.tolist()
    print(f"\nGINI (top {n_top}, {target_name}):")
    print(topn)
    return topn

sel_gini_uts  = gini_select(X_tr_uts_i,  y_tr_uts, 26, "UTS")
sel_gini_elon = gini_select(X_tr_el_i,   y_tr_el,   26, "Elongation")

# Reconstructig dataframes of the imputed train/test splits
X_tr_uts_imp = pd.DataFrame(
    imp_uts.transform(X_tr_uts),
    columns=X.columns,
    index=X_tr_uts.index
)
X_te_uts_imp = pd.DataFrame(
    imp_uts.transform(X_te_uts),
    columns=X.columns,
    index=X_te_uts.index
)
X_tr_el_imp = pd.DataFrame(
    imp_el.transform(X_tr_el),
    columns=X.columns,
    index=X_tr_el.index
)
X_te_el_imp = pd.DataFrame(
    imp_el.transform(X_te_el),
    columns=X.columns,
    index=X_te_el.index
)

# Choosing final features 
final_feats_uts  = sel_gini_uts
final_feats_elon = sel_gini_elon

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    "XGBoost":      xgb.XGBRegressor(n_estimators=100, random_state=0, verbosity=0),
    "GBDT":         GradientBoostingRegressor(n_estimators=100, random_state=0)
}

def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    r2   = r2_score(yte, ypred)
    rmse = np.sqrt(mean_squared_error(yte, ypred))
    print(f"{name:12s} → R² = {r2:.3f},   RMSE = {rmse:.3f}")

# Fitting and reporting for UTS
print("\n=== UTS (held-out 15%) ===")
for nm, mdl in models.items():
    evaluate(nm, mdl,
             X_tr_uts_imp[final_feats_uts],
             y_tr_uts,
             X_te_uts_imp[final_feats_uts],
             y_te_uts)


print("\n=== Elongation (held-out 15%) ===")
for nm, mdl in models.items():
    evaluate(nm, mdl,
             X_tr_el_imp[final_feats_elon],
             y_tr_el,
             X_te_el_imp[final_feats_elon],
             y_te_el)


