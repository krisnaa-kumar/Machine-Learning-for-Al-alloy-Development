#!/usr/bin/env python3
# multiobj_bayes.py
# Multi-objective Bayesian optimisation (ParEGO) to propose new Al alloys
# using controllable inputs (composition + processing). BO uses a GP for the
# acquisition model; XGBoost provides property predictions for candidates.
# Recommended to check report for detialed theory and equations behind code

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



FILE = "Individual Project Dataset-1.xlsx"
df_data  = pd.read_excel(FILE, sheet_name="My data", engine="openpyxl")
df_feats = pd.read_excel(FILE, sheet_name="element descriptors", engine="openpyxl")  # not used for BO

df_data.columns  = df_data.columns.str.strip()
df_feats.columns = df_feats.columns.str.strip()

mask = df_data["UTS (MPa)"].notna() & df_data["Elongation"].notna()
df_data = df_data.loc[mask].reset_index(drop=True)


composition_cols = [
    "Si","Fe","Cu","Be","Ag","Bi","Pb","Zn","Mn","Mg","Sn","Ti","V",
    "Mo","Ni","Ce","Co","Cr","Li","Sc","Sr","Zr","Al"  # note: 'Al' will be derived later
]
process_cols = [
    "Extrusion ratio","Extrusion speed (mm/s)","Extrusion Temp",
    "Solution_Temp (℃)","Solution_Time (h)","Quench_Temp (℃)",
    "Aging_Temp (℃)","Aging_Time (h)"
]


df_data[composition_cols] = df_data[composition_cols].astype(float).fillna(0.0)
df_data[process_cols]     = df_data[process_cols].astype(float)

# Targets for the two objectives
y_uts  = df_data["UTS (MPa)"].astype(float).values
y_elong= df_data["Elongation"].astype(float).values

X_full = df_data[composition_cols + process_cols].copy()

# Bayesian Optimisation search space uses non-Al + processing. Al is computed as remainder to 100%
non_al_cols = [c for c in composition_cols if c != "Al"]
p_valid = df_data[process_cols].clip(lower=0)     
if p_valid.isna().all(axis=0).any():
    missing_cols = p_valid.columns[p_valid.isna().all()].tolist()
    raise ValueError(f"No valid (>=0) observations for process columns: {missing_cols}")

lb = np.r_[df_data[non_al_cols].min().values, p_valid.min().values]
ub = np.r_[df_data[non_al_cols].max().values, p_valid.max().values]
X_red = df_data[non_al_cols + process_cols].copy()   # reduced inputs used by GP for acquisition

# Just for sanity checks and debugging, printing the feasible region the BO will respect
print("\n--- TRAINING BOUNDS (non-Al composition + process) ---")
print("Non-Al mins:\n", df_data[non_al_cols].min(numeric_only=True))
print("Non-Al maxs:\n", df_data[non_al_cols].max(numeric_only=True))
print("\nProcess mins (>=0 considered valid):\n", p_valid.min(numeric_only=True))
print("Process maxs:\n", p_valid.max(numeric_only=True))

# Use robust percentile bounds to avoid extreme corners and encourage safe exploration
USE_PERCENTILE_BOUNDS = True
if USE_PERCENTILE_BOUNDS:
    comp_lo = df_data[non_al_cols].quantile(0.05, numeric_only=True).values
    comp_hi = df_data[non_al_cols].quantile(0.95, numeric_only=True).values
    proc_lo = p_valid.quantile(0.05, numeric_only=True).values
    proc_hi = p_valid.quantile(0.95, numeric_only=True).values
    lb = np.r_[comp_lo, proc_lo]
    ub = np.r_[comp_hi, proc_hi]
    print("\nUsing robust (5–95%) bounds for BO.\n")


# Helper maps a reduced vector (non-Al + process) to full point. Enforces Al mass balance and clips process to bounds
def project_to_full(z):
    """
    z: array [len(non_al)+len(process)]
    returns full vector in order composition_cols + process_cols
    Enforces sum(composition)=100 by setting Al = 100 - sum(others),
    and clips to observed bounds of Al.
    """
    z = np.asarray(z, dtype=float)
    other = z[:len(non_al_cols)].copy()
    proc  = z[len(non_al_cols):].copy()

    # Keeping non-Al elements within observed limits 
    other = np.clip(other, X_red[non_al_cols].min().values, X_red[non_al_cols].max().values)

    others_sum = other.sum()
    al_min = df_data["Al"].min()
    al_max = df_data["Al"].max()
    max_others = 100.0 - al_min

    if others_sum > max_others:
        scale = max_others / (others_sum + 1e-12)
        other *= scale
        others_sum = other.sum()

    al = 100.0 - others_sum
    al = float(np.clip(al, al_min, al_max))

    full_comps = dict(zip(non_al_cols, other))
    full_comps["Al"] = al

    full = np.array([full_comps[c] for c in composition_cols] + list(proc), dtype=float)

    p_lb = p_valid.min().values
    p_ub = p_valid.max().values
    full[len(composition_cols):] = np.clip(full[len(composition_cols):], p_lb, p_ub)


    return full


# Training strong property predictors (XGB)

# Hold-out split for sanity checks of the surrogates
X_train, X_test, y1_tr, y1_te, y2_tr, y2_te = train_test_split(
    X_full.values, y_uts, y_elong, test_size=0.15, random_state=42
)

# Median imputation used for robustness on small data
imp = SimpleImputer(strategy="median").fit(X_train)
Xtr = imp.transform(X_train); Xte = imp.transform(X_test)

# Tuned XGB settings: shallow trees + shrinkage + subsampling -> good bias–variance on small n
xgb_uts = XGBRegressor(
    objective="reg:squarederror", random_state=42,
    n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.6
).fit(Xtr, y1_tr)
xgb_el  = XGBRegressor(
    objective="reg:squarederror", random_state=42,
    n_estimators=400, max_depth=4, learning_rate=0.1, subsample=0.8
).fit(Xtr, y2_tr)

# Wrapper used by BO: takes a full point and returns the two predicted properties
def predict_props(full_point_array):
    Xp = imp.transform(full_point_array.reshape(1, -1))
    uts = xgb_uts.predict(Xp).item()
    el  = xgb_el.predict(Xp).item()
    return float(uts), float(el)


# ParEGO MOBO using a GP acquisition model
# Observed set in reduced space (non-Al + process) and corresponding objective values
Z_obs_raw = df_data[non_al_cols + process_cols].to_numpy(copy=True)              
F_obs = np.column_stack([y_uts, y_elong])  

# Standardising inputs for the GP 
gp_imp   = SimpleImputer(strategy="median")
Z_obs    = gp_imp.fit_transform(Z_obs_raw)     
x_scaler = StandardScaler().fit(Z_obs)
Zs_obs   = x_scaler.transform(Z_obs)

# GP : Matern(v=2.5) + white noise -> robust to moderate non-smoothness, multiple restarts for stability.
# Check report for my information 
def make_gp(X, y):
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]),
                                          length_scale_bounds=(1e-2, 1e4), nu=2.5) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=0, n_restarts_optimizer=5)
    gp.fit(X, y)
    return gp

# Expected Improvement for a minimised scalar objective (ParEGO produces a scalarised y)
def expected_improvement_min(gp, Xcand, y_best, xi=0.01):
    mu, sigma = gp.predict(Xcand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = (y_best - mu - xi)
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# ParEGO scalarisation: random weights on the simplex + augmented Chebyshev term encourage balanced improvements
# Check report for my information and theory
def parego_scalarize(F, w, rho=0.05):
    # normalising each objective to [0,1] using current observed mins/maxes
    mins = F.min(axis=0); maxs = F.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-9)
    Z = (F - mins) / denom           
    y = 1.0 - Z                       # converting to minimisation
    tcheb = np.max(w * y, axis=1)
    aug   = tcheb + rho * np.sum(w * y, axis=1)
    return aug.reshape(-1, 1)

# Candidate sampler: uniform within the feasible box (lb/ub)
rng = np.random.default_rng(123)

def sample_reduced(n):
    U = rng.random((n, len(non_al_cols) + len(process_cols)))
    return lb + U * (ub - lb)

# Pareto mask to keep non-dominated points in objective space (maximise both UTS and elongation)
def pareto_mask(F):  
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        # any j dominates i?
        if np.any((F >= F[i]).all(axis=1) & (F > F[i]).any(axis=1) & (np.arange(n) != i)):
            keep[i] = False
    return keep

# --- BO loop ---
T        = 30        # iterations
q        = 5 
N_CAND = 4000        # candidates per iteration (diverse w using random weights)
batch    = []
all_cands= []

for t in range(T):
    # Random weights over the 2 objectives -> different trade-offs each iteration
    w = rng.dirichlet(alpha=np.ones(2), size=1).reshape(-1)

    # Scalarise current observations with ParEGO and fit a GP to that scalar objective
    y_scalar = parego_scalarize(F_obs, w)      
    gp = make_gp(Zs_obs, y_scalar.ravel())

    # Proposing points by maximising EI over many random samples
    Zs_cand = x_scaler.transform(sample_reduced(N_CAND))
    ei = expected_improvement_min(gp, Zs_cand, y_best=float(y_scalar.min()))
    top_idx = np.argsort(ei)[-q:]

    for idx in top_idx:
        # Map reduced -> full, enforced Al mass balance 
        z_red = x_scaler.inverse_transform(Zs_cand[idx:idx+1])[0]
        x_full = project_to_full(z_red)

        # Scoring with trained surrogates (XGB) to obtain objective values
        uts_pred, el_pred = predict_props(x_full)

        # Recording candidate with predictions and ParEGO weights 
        cand = {
            **{c: x_full[i] for i, c in enumerate(composition_cols)},
            **{c: x_full[len(composition_cols)+j] for j, c in enumerate(process_cols)},
            "pred_UTS": uts_pred,
            "pred_Elongation": el_pred,
            "iter": t+1,
            "w1": float(w[0]),
            "w2": float(w[1]),
        }
        all_cands.append(cand)

        # Updating the observed set so the GP learns from newly proposed predictions
        Z_obs_raw = np.vstack([Z_obs_raw, z_red])     
        Z_obs     = gp_imp.fit_transform(Z_obs_raw)   # re-fit imputer as dataset grows
        Zs_obs    = x_scaler.fit_transform(Z_obs)     # and rescale
        F_obs     = np.vstack([F_obs, [uts_pred, el_pred]])   # update scaler as the space grows

cand_df = pd.DataFrame(all_cands)

# Extracting predicted Pareto set among proposed candidates 
F = cand_df[["pred_UTS", "pred_Elongation"]].values
mask = pareto_mask(F)
pareto_df = cand_df.loc[mask].sort_values(["pred_UTS", "pred_Elongation"], ascending=False)

# Novelty: Mahalanobis distance from the training cloud plot
# measures distribution shift from the training data
Z_train = gp_imp.transform(df_data[non_al_cols + process_cols].to_numpy(copy=True))  

# Ledoit–Wolf shrinkage - check report for theory
lw = LedoitWolf().fit(Z_train)
mu = lw.location_                      # mean vector 
prec = lw.precision_                   # inverse covariance 
d_train = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z_train - mu, prec, Z_train - mu), 0.0))

Z_cand = gp_imp.transform(cand_df[non_al_cols + process_cols].to_numpy(copy=True))
d_cand = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z_cand - mu, prec, Z_cand - mu), 0.0))

# Percentile score indicates how far a candidate is from the training cloud 
d_train_sorted = np.sort(d_train)
pctl = 100.0 * np.searchsorted(d_train_sorted, d_cand, side="right") / len(d_train_sorted)

cand_df["mah_dist"] = d_cand
cand_df["mah_pctl_vs_train"] = pctl
cand_df["novel_outside95"] = cand_df["mah_pctl_vs_train"] > 95.0

pareto_df = cand_df.loc[pareto_df.index]

# again sanity checks for debugging purposes
def _audit(cand_df):
    comp_sum = cand_df[composition_cols].sum(axis=1)
    print("Comp sum min/median/max:", comp_sum.min(), comp_sum.median(), comp_sum.max())
    off = (comp_sum.sub(100).abs() > 0.5).sum()
    print(f"Compositions off by >0.5% from 100: {off} rows")

    # checking processing inside training bounds
    train_proc_min = p_valid.min(numeric_only=True)
    train_proc_max = p_valid.max(numeric_only=True)
    cand_proc_min  = cand_df[process_cols].min(numeric_only=True)
    cand_proc_max  = cand_df[process_cols].max(numeric_only=True)
    print("\nPROCESS bounds check:")
    print("  training min:\n", train_proc_min)
    print("  training max:\n", train_proc_max)
    print("  candidate min:\n", cand_proc_min)
    print("  candidate max:\n", cand_proc_max)

    for c in process_cols:
        assert cand_proc_min[c] >= train_proc_min[c] - 1e-9, f"{c} below training min"
        assert cand_proc_max[c] <= train_proc_max[c] + 1e-9, f"{c} above training max"

_audit(cand_df)

# Ranking Pareto points with distance to utopia (1,1) on normalised objectives
def rank_pareto_points(pareto_df, cand_df, k=5):
    u_all = cand_df["pred_UTS"]; e_all = cand_df["pred_Elongation"]
    u_min, u_max = float(u_all.min()), float(u_all.max())
    e_min, e_max = float(e_all.min()), float(e_all.max())
    u_n = (pareto_df["pred_UTS"] - u_min) / (u_max - u_min + 1e-9)
    e_n = (pareto_df["pred_Elongation"] - e_min) / (e_max - e_min + 1e-9)
    utopia_dist = np.sqrt((1.0 - u_n)**2 + (1.0 - e_n)**2)
    ranked = pareto_df.assign(utopia_dist=utopia_dist).sort_values("utopia_dist")
    ranked = ranked.head(k).copy()
    ranked.insert(0, "ID", [f"P{i+1}" for i in range(len(ranked))])
    return ranked

top_k = 5
top_pareto = rank_pareto_points(pareto_df, cand_df, k=top_k)

cols_table = (["ID", "pred_UTS", "pred_Elongation", "mah_pctl_vs_train", "utopia_dist"]
              + composition_cols + process_cols + ["iter", "w1", "w2"])
(top_pareto[cols_table]
 .sort_values("ID")
 .to_csv("bo_top_pareto_table.csv", index=False))
print(f"Saved top-{len(top_pareto)} Pareto table -> bo_top_pareto_table.csv")


def plot_pareto_scatter(cand_df, pareto_df, fname="bo_pareto_scatter.png"):
    """
    Scatter of all candidates (small), Pareto points highlighted (large),
    UTS on y-axis, Elongation on x-axis — like the paper’s figure style.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)

    # All candidates
    ax.scatter(cand_df["pred_Elongation"], cand_df["pred_UTS"],
               s=18, alpha=0.65, label="Candidates")

    # Pareto front highlighted
    ax.scatter(pareto_df["pred_Elongation"], pareto_df["pred_UTS"],
               s=60, marker="o", edgecolor="k", linewidth=0.6, label="Pareto set")

    ax.set_xlabel("Predicted Elongation (%)", fontsize=10)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=10)
    ax.set_title("Proposed alloys: UTS vs Elongation", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_with_novelty(cand_df, pareto_df, fname="bo_pareto_scatter_novelty.png"):
    """
    Same scatter but colour by Mahalanobis distance percentile vs training.
    Pareto points outlined.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)

    sc = ax.scatter(cand_df["pred_Elongation"], cand_df["pred_UTS"],
                    c=cand_df["mah_pctl_vs_train"], s=20, alpha=0.85)

    ax.scatter(pareto_df["pred_Elongation"], pareto_df["pred_UTS"],
               s=70, facecolor="none", edgecolor="k", linewidth=0.9, label="Pareto set")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance from training cloud (percentile)", fontsize=9)

    ax.set_xlabel("Predicted Elongation (%)", fontsize=10)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=10)
    ax.set_title("Proposed alloys: UTS vs Elongation (novelty)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _add_training_background(ax, y_elong, y_uts):
    # faint background of the original dataset
    ax.scatter(y_elong, y_uts, s=10, alpha=0.18, color="#6f6f6f", label="Training data")

def plot_pareto_with_background(cand_df, pareto_df, y_elong, y_uts,
                                top_pareto, fname="bo_pareto_with_training.png"):
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=150)

    _add_training_background(ax, y_elong, y_uts)

    # all proposed candidates
    ax.scatter(cand_df["pred_Elongation"], cand_df["pred_UTS"],
               s=18, alpha=0.65, label="Candidates")

    # Pareto set highlighted
    ax.scatter(pareto_df["pred_Elongation"], pareto_df["pred_UTS"],
               s=60, marker="o", edgecolor="k", linewidth=0.6, facecolor="tab:orange",
               label="Pareto set")

    # labelling the top Pareto picks
    for _, r in top_pareto.iterrows():
        ax.annotate(r["ID"],
                    xy=(r["pred_Elongation"], r["pred_UTS"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("Predicted Elongation (%)", fontsize=10)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)

def plot_pareto_novelty_with_background(cand_df, pareto_df, y_elong, y_uts,
                                        top_pareto, fname="bo_pareto_novelty_with_training.png"):
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=150)

    _add_training_background(ax, y_elong, y_uts)

    sc = ax.scatter(cand_df["pred_Elongation"], cand_df["pred_UTS"],
                    c=cand_df["mah_pctl_vs_train"], s=20, alpha=0.85)

    ax.scatter(pareto_df["pred_Elongation"], pareto_df["pred_UTS"],
               s=70, facecolor="none", edgecolor="k", linewidth=0.9, label="Pareto set")

    for _, r in top_pareto.iterrows():
        ax.annotate(r["ID"],
                    xy=(r["pred_Elongation"], r["pred_UTS"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance from training cloud (percentile)", fontsize=9)

    ax.set_xlabel("Predicted Elongation (%)", fontsize=10)
    ax.set_ylabel("Predicted UTS (MPa)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)



cand_df.to_csv("bo_candidates_all-1.csv", index=False)
pareto_df.to_csv("bo_candidates_pareto-1.csv", index=False)

# Plots
plot_pareto_scatter(cand_df, pareto_df, fname="bo_pareto_scatter.png")
plot_pareto_with_novelty(cand_df, pareto_df, fname="bo_pareto_scatter_novelty.png")
print("Saved plots:")
print("  - bo_pareto_scatter.png")
print("  - bo_pareto_scatter_novelty.png")

print(f"Generated {len(cand_df)} candidate points over {T} iterations.")
print(f"Pareto set size: {pareto_df.shape[0]}")
print("Saved:")
print("  - bo_candidates_all-1.csv")
print("  - bo_candidates_pareto-1.csv")

plot_pareto_with_background(cand_df, pareto_df, y_elong, y_uts, top_pareto,
                            fname="bo_pareto_with_training.png")
plot_pareto_novelty_with_background(cand_df, pareto_df, y_elong, y_uts, top_pareto,
                                    fname="bo_pareto_novelty_with_training.png")
print("Also saved:")
print("  - bo_pareto_with_training.png")
print("  - bo_pareto_novelty_with_training.png")

comp_sum = cand_df[composition_cols].sum(axis=1)
print("Comp sum: min/median/max =",
      comp_sum.min(), comp_sum.median(), comp_sum.max())
off = (comp_sum.sub(100).abs() > 0.5).sum()
print(f"Compositions off by >0.5% from 100: {off} rows")

print("Al range among candidates:", cand_df["Al"].min(), cand_df["Al"].max())
print("Processing mins (should be >= lower bounds):")
print(cand_df[process_cols].min())

print("Training UTS range:", y_uts.min(), y_uts.max())
print("Training Elongation range:", y_elong.min(), y_elong.max())
print("Candidates UTS range:", cand_df["pred_UTS"].min(), cand_df["pred_UTS"].max())
print("Candidates Elongation range:", cand_df["pred_Elongation"].min(), cand_df["pred_Elongation"].max())



