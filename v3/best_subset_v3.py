#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best-subset selection for v3 using NB2 throughout.
Fits all 2^15 - 1 = 32,767 NB2 models across all subsets of the 15
candidate predictors. Selects the best by AIC and by BIC.

Run from project root:  python v3/best_subset_v3.py
Output saved to v3 folder.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from itertools import combinations

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = "v3"

# ── Data prep (identical to v3) ───────────────────────────────────────────────
df_raw = pd.read_csv("stats.csv")
df_raw.rename(columns={"last_name, first_name": "player_name"}, inplace=True)
for col in ["on_base_percent","on_base_plus_slg","xba","xslg","woba","xwoba",
            "xobp","xiso","xwobacon","xbacon","sprint_speed"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.strip('"'), errors="coerce")
df_raw = df_raw[df_raw["year"] <= 2025].copy()

LAG_COLS = ["k_percent","bb_percent","xslg","xwobacon","exit_velocity_avg",
            "barrel_batted_rate","solidcontact_percent","hard_hit_percent",
            "avg_best_speed","avg_hyper_speed","whiff_percent","swing_percent","sprint_speed"]

cur  = df_raw[["player_id","player_name","year","player_age","pa","home_run"]].copy()
prev = df_raw[["player_id","year"] + LAG_COLS].copy()
prev.rename(columns={c: c+"_lag" for c in LAG_COLS}, inplace=True)
prev["year"] += 1
df = cur.merge(prev, on=["player_id","year"], how="inner")
df["age2"] = df["player_age"]**2
lag_lag = [c+"_lag" for c in LAG_COLS]
df = df.dropna(subset=lag_lag + ["pa","home_run"]).copy()
df = df[df["pa"] >= 300].copy()
df.reset_index(drop=True, inplace=True)
df["log_pa"] = np.log(df["pa"])
n_obs = len(df)

print(f"Dataset: {n_obs:,} player-seasons")

# ── Candidate predictors (15 total) ──────────────────────────────────────────
CANDIDATES = lag_lag + ["player_age", "age2"]
n_cands  = len(CANDIDATES)
n_subsets = 2**n_cands - 1
print(f"Candidates: {n_cands}  |  Subsets: 2^{n_cands} - 1 = {n_subsets:,}")


def fit_nb(variables):
    """Fit NB2 on df and return (aic, std_bic, model)."""
    formula = "home_run ~ " + " + ".join(variables)
    m = smf.negativebinomial(formula, data=df, offset=df["log_pa"]).fit(
        disp=False, method="bfgs", maxiter=200)
    bic = -2.0 * m.llf + len(m.params) * np.log(n_obs)
    return m.aic, bic, m


# ── Exhaustive search ─────────────────────────────────────────────────────────
best_aic_val,  best_aic_vars  = np.inf, []
best_bic_val,  best_bic_vars  = np.inf, []

# Track best AIC and BIC at each subset size for profile plot
profile_aic = {}
profile_bic = {}

t0    = time.perf_counter()
count = 0

print(f"\nFitting {n_subsets:,} NB2 models …\n")

for size in range(1, n_cands + 1):
    size_best_aic = np.inf
    size_best_bic = np.inf

    for subset in combinations(CANDIDATES, size):
        vlist = list(subset)
        try:
            aic, bic, _ = fit_nb(vlist)
        except Exception:
            count += 1
            continue

        if aic < best_aic_val:
            best_aic_val, best_aic_vars = aic, vlist[:]
        if bic < best_bic_val:
            best_bic_val, best_bic_vars = bic, vlist[:]

        size_best_aic = min(size_best_aic, aic)
        size_best_bic = min(size_best_bic, bic)
        count += 1

        if count % 1000 == 0:
            elapsed = time.perf_counter() - t0
            rate    = count / elapsed
            remain  = (n_subsets - count) / rate
            pct     = 100 * count / n_subsets
            print(f"  {count:>6}/{n_subsets:,}  ({pct:4.1f}%)  "
                  f"elapsed {elapsed/60:.1f} min  "
                  f"~{remain/60:.1f} min remaining  "
                  f"best AIC so far: {best_aic_val:.2f}")

    profile_aic[size] = size_best_aic
    profile_bic[size] = size_best_bic
    print(f"  Size {size:>2}: best AIC = {size_best_aic:.2f},  best BIC = {size_best_bic:.2f}")

elapsed_total = time.perf_counter() - t0
print(f"\nDone. {n_subsets:,} models in {elapsed_total/60:.1f} min "
      f"({elapsed_total/n_subsets*1000:.1f} ms/model)")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nBest AIC = {best_aic_val:.2f}  ({len(best_aic_vars)} predictors)")
print(f"  {best_aic_vars}")
print(f"\nBest BIC = {best_bic_val:.2f}  ({len(best_bic_vars)} predictors)")
print(f"  {best_bic_vars}")

# Fit and summarise both final models
print("\n── Best-AIC NB2 summary ────────────────────────────────────────────────")
_, _, m_aic = fit_nb(best_aic_vars)
print(m_aic.summary())
print("\nFitted formula — best-AIC NB2:")
print("  log E[HR_t] = log(PA_t)")
for name, coef in {k: v for k, v in m_aic.params.items() if k != "alpha"}.items():
    print(f"    + ({coef:+.5f}) · {name}")
print(f"  alpha = {m_aic.params['alpha']:.4f}")

print("\n── Best-BIC NB2 summary ────────────────────────────────────────────────")
_, _, m_bic = fit_nb(best_bic_vars)
print(m_bic.summary())
print("\nFitted formula — best-BIC NB2:")
print("  log E[HR_t] = log(PA_t)")
for name, coef in {k: v for k, v in m_bic.params.items() if k != "alpha"}.items():
    print(f"    + ({coef:+.5f}) · {name}")
print(f"  alpha = {m_bic.params['alpha']:.4f}")

# ── Best-subset profile plot ──────────────────────────────────────────────────
sizes = list(range(1, n_cands + 1))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Best-subset profile — NB2 (v3,  PA≥300,  15 candidates)", fontsize=10)

axes[0].plot(sizes, [profile_aic[s] for s in sizes], "o-", color="steelblue", lw=1.5)
axes[0].axvline(len(best_aic_vars), color="red", ls="--", lw=1.2,
                label=f"Best k={len(best_aic_vars)}")
axes[0].set_xlabel("# predictors"); axes[0].set_ylabel("AIC")
axes[0].set_title("AIC by subset size"); axes[0].legend(fontsize=8)

axes[1].plot(sizes, [profile_bic[s] for s in sizes], "o-", color="darkorange", lw=1.5)
axes[1].axvline(len(best_bic_vars), color="red", ls="--", lw=1.2,
                label=f"Best k={len(best_bic_vars)}")
axes[1].set_xlabel("# predictors"); axes[1].set_ylabel("BIC")
axes[1].set_title("BIC by subset size"); axes[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "best_subset_profile_v3.png"), bbox_inches="tight")
plt.close()
print(f"\n-> {OUT_DIR}/best_subset_profile_v3.png")

# Save results to text file
out_path = os.path.join(OUT_DIR, "best_subset_results_v3.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"Best-subset NB2 results (v3, PA>=300, 15 candidates)\n")
    f.write(f"{'='*60}\n")
    f.write(f"Best AIC = {best_aic_val:.4f}  ({len(best_aic_vars)} predictors)\n")
    f.write(f"  {best_aic_vars}\n\n")
    f.write(f"Best BIC = {best_bic_val:.4f}  ({len(best_bic_vars)} predictors)\n")
    f.write(f"  {best_bic_vars}\n\n")
    f.write("Profile (best AIC/BIC by subset size):\n")
    for s in sizes:
        f.write(f"  k={s:>2}: AIC={profile_aic[s]:.2f},  BIC={profile_bic[s]:.2f}\n")
print(f"-> {OUT_DIR}/best_subset_results_v3.txt")
