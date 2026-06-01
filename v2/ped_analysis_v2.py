#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson (and Negative Binomial) GLM Analysis — v2 (revised predictor set)
==========================================================================
Changes vs v1 (ped_analysis.py):

  1. barrel_batted_rate_lag  → log1p(barrel_batted_rate_lag)
     Right-skewed (skew 0.76) with a nonlinear scatter vs log(HR rate);
     log1p linearises the relationship and compresses the heavy right tail.

  2. x-stat block reduced to xslg_lag + xwobacon_lag only.
     v1 included xba, xslg, xiso, xwoba, xobp, xwobacon, xbacon — all
     heavily correlated, producing large opposing coefficients (classic VIF
     inflation). We keep:
       * xslg  — best single-stat summary of extra-base-hit contact quality
       * xwobacon — quality of contact when the ball is put in play
     and drop xba, xiso, xwoba, xobp, xbacon.

Run:   python ped_analysis_v2.py
Output: console text  +  PNG figures with _v2 suffix
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import poisson as sp_poisson, nbinom as sp_nbinom
import warnings
import sys
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows so Unicode arrows/box-drawing chars print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING AND CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("1. DATA LOADING AND CLEANING")
print("=" * 72)

df_raw = pd.read_csv("stats.csv")
df_raw.rename(columns={"last_name, first_name": "player_name"}, inplace=True)

# Several stat columns are stored as quoted strings (e.g. ".350") — strip and cast.
QUOTED_COLS = [
    "on_base_percent", "on_base_plus_slg",
    "xba", "xslg", "woba", "xwoba", "xobp",
    "xiso", "xwobacon", "xbacon", "sprint_speed",
]
for col in QUOTED_COLS:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(
            df_raw[col].astype(str).str.strip('"'), errors="coerce"
        )

# Drop incomplete 2026 season.
df_raw = df_raw[df_raw["year"] <= 2025].copy()

print(f"  Rows : {len(df_raw):,}")
print(f"  Years: {df_raw['year'].min()}–{df_raw['year'].max()}")
print(f"  Players: {df_raw['player_id'].nunique():,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VARIABLE EXCLUSION RATIONALE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("2. VARIABLE EXCLUSION RATIONALE")
print("=" * 72)

print("""
EXCLUDED — and why
──────────────────────────────────────────────────────────────────────────────
  player_name, player_id, year
    Identifiers / time-index only; carry no physical information.

  home_run
    The RESPONSE variable.

  ab, hit, single, double, triple
    Raw counting stats that scale with PA volume and are determined by
    the same at-bat outcomes we are trying to explain. Including them
    would create near-tautological predictors of HR count.

  strikeout, walk
    Replaced by their RATE equivalents k_percent / bb_percent, which
    are independent of PA volume.

  on_base_plus_slg (OPS), woba, on_base_percent
    Realized composite/rate stats that directly incorporate the home-run
    events we are modelling.  Slugging percentage, for example, weights
    a home run at 4×; including it would put the response on the RHS.

  avg_swing_speed, blasts_contact
    Only available for 2023–2025 (75.5 % null across the full dataset).
    Including them as lagged predictors would restrict the sample to
    2024–2025, losing >90 % of the data.

KEPT as lagged predictors (prior-year stats, except age which is current)
──────────────────────────────────────────────────────────────────────────────
  Statcast "expected" stats: xslg, xwobacon
    (v2: reduced from 7 to 2 to eliminate multicollinearity; xslg captures
    extra-base power, xwobacon captures contact quality independent of walks)
  Contact quality   : log1p(barrel_batted_rate) [v2: log-transformed],
                      exit_velocity_avg, hard_hit_percent,
                      solidcontact_percent, avg_best_speed, avg_hyper_speed
  Plate discipline  : k_percent, bb_percent
  Swing behaviour   : whiff_percent, swing_percent
  Athleticism       : sprint_speed
  Age curve control : player_age, player_age² (current year, not lagged)

AGE MODELLING CHOICE
──────────────────────────────────────────────────────────────────────────────
  We use age + age² (quadratic) rather than age × stat interactions.
  The quadratic form captures the smooth rise-and-fall of the MLB power
  arc with only 2 extra degrees of freedom; age interactions with each
  predictor would add ~18 df and risk overfitting. The stepwise search
  can add age interactions if they genuinely help.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD LAGGED-PREDICTOR DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("3. BUILD LAGGED-PREDICTOR DATASET")
print("=" * 72)

# Predictors we will lag (use year t-1 values to predict year t HR count).
# avg_swing_speed and blasts_contact are excluded due to sparsity (see §2).
LAG_COLS = [
    "k_percent", "bb_percent",
    # x-stat block trimmed to two non-collinear representatives (v2 change)
    "xslg", "xwobacon",
    "exit_velocity_avg", "barrel_batted_rate",
    "solidcontact_percent", "hard_hit_percent",
    "avg_best_speed", "avg_hyper_speed",
    "whiff_percent", "swing_percent", "sprint_speed",
]

# Current-year outcome and exposure
cur = df_raw[[
    "player_id", "player_name", "year", "player_age", "pa", "home_run"
]].copy()

# Prior-year predictors — rename so we know they are lagged
prev = df_raw[["player_id", "year"] + LAG_COLS].copy()
prev.rename(columns={c: c + "_lag" for c in LAG_COLS}, inplace=True)
prev["year"] += 1   # shift so the prior-year row aligns with the current year

df = cur.merge(prev, on=["player_id", "year"], how="inner")

# Quadratic age term for the nonlinear career arc
df["age2"] = df["player_age"] ** 2

# Drop rows where any lagged predictor is missing or PA is too small.
# Requiring ≥ 100 PA removes part-time/injured seasons that add noise.
LAG_LAG = [c + "_lag" for c in LAG_COLS]
df = df.dropna(subset=LAG_LAG + ["pa", "home_run"]).copy()
df = df[df["pa"] >= 100].copy()
df.reset_index(drop=True, inplace=True)

# Named offset column: log(PA) so the model is on a per-PA rate scale
df["log_pa"] = np.log(df["pa"])
# Empirical HR rate (used in plots)
df["hr_rate"] = df["home_run"] / df["pa"]

# v2: log1p-transform barrel_batted_rate_lag to linearise its right-skewed
# relationship with log(HR rate) (skew 0.76, nonlinear scatter in v1 fig 2).
# log1p(x) = log(1+x) handles x=0 safely and compresses the tail.
df["log_barrel_lag"] = np.log1p(df["barrel_batted_rate_lag"])

print(f"  Lagged dataset: {len(df):,} player-seasons "
      f"({df['year'].min()}–{df['year'].max()})")
print(f"  Players        : {df['player_id'].nunique():,}")
print(f"  Mean HR        : {df['home_run'].mean():.2f}  |  "
      f"Median: {df['home_run'].median():.0f}  |  "
      f"Max: {df['home_run'].max():.0f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OVERDISPERSION CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("4. OVERDISPERSION CHECK")
print("=" * 72)

hr_mean = df["home_run"].mean()
hr_var  = df["home_run"].var()
raw_disp = hr_var / hr_mean

print(f"""
Poisson assumes Var(Y) = E[Y].  If Var >> mean, we have overdispersion
and the Negative Binomial (which adds a free variance parameter) is more
appropriate.

  Overall HR mean        = {hr_mean:.3f}
  Overall HR variance    = {hr_var:.3f}
  Variance / mean ratio  = {raw_disp:.2f}

A ratio of {raw_disp:.1f} far exceeds 1.0, indicating strong raw overdispersion.
This partly reflects between-player heterogeneity in PA (some players have
600 PA, others 100).  After conditioning on lagged contact-quality metrics,
the *residual* dispersion will be lower — but likely still > 1.

We will:
  (a) Fit a Poisson GLM (our primary M_0 as requested) and check the
      residual deviance / df ratio.
  (b) Formally test for overdispersion using the Cameron & Trivedi (1990)
      auxiliary regression test.
  (c) Fit a Negative Binomial (NB2) model and compare via AIC/BIC.
""")

# Cameron & Trivedi auxiliary regression test for overdispersion
# H₀ (Poisson): E[(y - μ)² / μ - 1] = 0
# We regress z = (y - μ̂)² / μ̂ - 1  on  μ̂  (or 1/μ̂).
# A significantly positive slope indicates overdispersion.
# We first need a Poisson fit to get μ̂.  Use the null (intercept-only) model.
null_formula = "home_run ~ 1"
m_null_pois = smf.glm(
    null_formula, data=df,
    family=sm.families.Poisson(),
    offset=df["log_pa"]
).fit(disp=0)

mu_null = m_null_pois.fittedvalues
ct_z = (df["home_run"] - mu_null) ** 2 / mu_null - 1
ct_result = sm.OLS(ct_z, sm.add_constant(mu_null)).fit()

print("  Cameron & Trivedi overdispersion test (on null Poisson model):")
print(f"    slope = {ct_result.params.iloc[1]:.4f},  "
      f"t = {ct_result.tvalues.iloc[1]:.3f},  "
      f"p = {ct_result.pvalues.iloc[1]:.4g}")
if ct_result.pvalues.iloc[1] < 0.05:
    print("    → Overdispersion confirmed. Negative Binomial is recommended.\n")
else:
    print("    → No significant overdispersion detected.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EXPLORATORY PLOTS OF PREDICTORS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("5. EXPLORATORY PLOTS")
print("=" * 72)

# Small epsilon so log(hr_rate) is defined for zero-HR seasons
EPS = 1e-4
df["log_hr_rate"] = np.log(df["hr_rate"] + EPS)

# ── 5a. Predictor histograms with skewness annotation ──────────────────────
# Include log_barrel_lag in the plots (replaces barrel_batted_rate_lag)
plot_vars = (
    [c for c in LAG_LAG if c != "barrel_batted_rate_lag"]
    + ["log_barrel_lag", "player_age"]
)
n_cols = 4
n_rows = int(np.ceil(len(plot_vars) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.8))
axes = axes.flatten()

for i, col in enumerate(plot_vars):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.82)
    label = col.replace("_lag", "").replace("_", " ")
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_ylabel("Count", fontsize=7)
    sk = float(stats.skew(data))
    # Flag heavy right-skew: a log transform might help
    colour = "darkred" if abs(sk) > 1.0 else "dimgray"
    ax.annotate(f"skew={sk:.2f}", xy=(0.65, 0.88), xycoords="axes fraction",
                fontsize=7, color=colour)

for j in range(len(plot_vars), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Distributions of lagged predictors\n"
    "(red skewness annotation = |skew| > 1 → consider log-transform)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("fig1_predictor_distributions_v2.png", bbox_inches="tight")
plt.close()
print("  → fig1_predictor_distributions_v2.png")

# ── 5b. Scatter: lagged predictor vs log(HR rate), with Pearson r ──────────
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.8))
axes2 = axes2.flatten()

scatter_cols = (
    [c for c in LAG_LAG if c != "barrel_batted_rate_lag"]
    + ["log_barrel_lag"]
)
corr_with_hr = {}
for i, col in enumerate(scatter_cols):
    ax = axes2[i]
    mask = df[[col, "log_hr_rate"]].notna().all(axis=1)
    x = df.loc[mask, col].values
    y = df.loc[mask, "log_hr_rate"].values
    ax.scatter(x, y, s=4, alpha=0.25, color="steelblue")
    if len(x) > 5:
        m, b = np.polyfit(x, y, 1)
        xr = np.array([x.min(), x.max()])
        ax.plot(xr, m * xr + b, color="red", lw=1.2)
        r_val = np.corrcoef(x, y)[0, 1]
        corr_with_hr[col.replace("_lag", "")] = r_val
        colour = "darkred" if abs(r_val) > 0.4 else "dimgray"
        ax.annotate(f"r={r_val:.2f}", xy=(0.05, 0.88), xycoords="axes fraction",
                    fontsize=7, color=colour)
    label = col.replace("_lag", "").replace("_", " ")
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_ylabel("log(HR rate)", fontsize=7)

for j in range(len(scatter_cols), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle(
    "Lagged predictor vs log(HR rate)  (red line = OLS trend, r = Pearson correlation)\n"
    "Red r-values indicate |r| > 0.4",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("fig2_predictor_vs_hr_rate_v2.png", bbox_inches="tight")
plt.close()
print("  → fig2_predictor_vs_hr_rate_v2.png")

# Print correlation ranking
corr_df = (
    pd.DataFrame.from_dict(corr_with_hr, orient="index", columns=["r"])
    .sort_values("r", ascending=False)
)
print("\n  Pearson correlations with log(HR rate) — ranked:")
for var, row in corr_df.iterrows():
    print(f"    {var:<30}  r = {row['r']:+.3f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INITIAL POISSON MODEL M_0 (PED-SENSITIVE PREDICTORS)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("6. INITIAL POISSON MODEL M_0 — PED-SENSITIVE PREDICTORS")
print("=" * 72)

print("""
Heuristic for M_0 variable selection
─────────────────────────────────────────────────────────────────────────────
Anabolic steroids and HGH increase lean muscle mass and neuromuscular
efficiency, which should manifest most directly in hard-contact metrics:

  barrel_batted_rate_lag  — the fraction of balls hit with ideal exit
                            velocity AND launch angle to produce extra-
                            base hits; arguably the single best Statcast
                            proxy for raw power output.
  hard_hit_percent_lag    — balls hit ≥ 95 mph EV; reflects sustained
                            strength throughout the zone.
  exit_velocity_avg_lag   — average exit velocity across all batted balls;
                            boosted directly by muscle mass and bat speed.
  xiso_lag                — expected isolated power (xSLG − xBA); captures
                            the extra-base-hit component of contact quality,
                            which is most tightly linked to HR potential.
  avg_best_speed_lag      — mean of the player's top-percentile exit velos;
                            proxy for peak raw power in the best swings.

  player_age, age2        — quadratic career arc control (age + age²).

These five metrics capture the physical dimensions most directly enhanced
by strength-boosting substances.  Statcast "expected" stats capture the
same signal from a different angle but are more collinear with each other,
so we prefer the raw contact-quality metrics here.
""")

# Predictors for M_0
M0_VARS = [
    "log_barrel_lag", "hard_hit_percent_lag",      # v2: log-barrel replaces raw barrel
    "exit_velocity_avg_lag", "xslg_lag", "avg_best_speed_lag",  # xslg replaces xiso
    "player_age", "age2",
]

m_pois_0 = smf.glm(
    "home_run ~ " + " + ".join(M0_VARS),
    data=df,
    family=sm.families.Poisson(),
    offset=df["log_pa"],
).fit(disp=0)

print("─── M_0 Poisson summary ───────────────────────────────────────────────")
print(m_pois_0.summary())

# Fitted formula with actual coefficient estimates
print("\nFitted formula for M_0:")
print("  log E[HR_t] = log(PA_t)")
intercept = m_pois_0.params["Intercept"]
print(f"    + ({intercept:+.5f})  [Intercept]")
for var in M0_VARS:
    coef = m_pois_0.params[var]
    print(f"    + ({coef:+.5f}) · {var}")

# Residual deviance / df — the Poisson goodness-of-fit diagnostic
disp_m0 = m_pois_0.deviance / m_pois_0.df_resid
print(f"\n  Residual deviance / df_resid = "
      f"{m_pois_0.deviance:.1f} / {m_pois_0.df_resid} = {disp_m0:.3f}")
if disp_m0 > 1.5:
    print("  ⚠  Ratio well above 1 — overdispersion present even after "
          "conditioning on predictors. NB model is preferred.")
else:
    print("  ✓  Ratio close to 1 — Poisson variance assumption is reasonable.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  M_0 DIAGNOSTICS (PLOTS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("7. M_0 DIAGNOSTICS")
print("=" * 72)

fitted_m0 = m_pois_0.fittedvalues
pearson_m0 = (df["home_run"] - fitted_m0) / np.sqrt(fitted_m0)
deviance_m0 = m_pois_0.resid_deviance

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))
fig3.suptitle("M_0 Diagnostics (Poisson GLM — PED-sensitive predictors)", fontsize=11)

# (a) Pearson residuals vs log(fitted)
ax = axes3[0, 0]
ax.scatter(np.log(fitted_m0), pearson_m0, s=5, alpha=0.35, color="steelblue")
ax.axhline(0, color="black", lw=0.8)
ax.axhline(2,  color="orange", lw=0.9, ls="--", label="±2")
ax.axhline(-2, color="orange", lw=0.9, ls="--")
ax.axhline(3,  color="red",    lw=0.9, ls=":",  label="±3")
ax.axhline(-3, color="red",    lw=0.9, ls=":")
ax.set_xlabel("log(fitted HRs)")
ax.set_ylabel("Pearson residuals")
ax.set_title("(a) Residuals vs Fitted")
ax.legend(fontsize=7)

# (b) Q-Q plot of Pearson residuals vs N(0,1)
ax = axes3[0, 1]
(osm, osr), (slope, intercept_qq, r_qq) = stats.probplot(pearson_m0, dist="norm")
ax.plot(osm, osr, "o", ms=3, alpha=0.35, color="steelblue")
ax.plot(osm, slope * np.array(osm) + intercept_qq, color="red", lw=1.2)
ax.set_xlabel("Theoretical quantiles  N(0,1)")
ax.set_ylabel("Sample quantiles")
ax.set_title("(b) Q-Q plot — Pearson residuals")

# (c) Scale-location (√|residuals| vs fitted)
ax = axes3[0, 2]
ax.scatter(np.log(fitted_m0), np.sqrt(np.abs(pearson_m0)),
           s=5, alpha=0.35, color="steelblue")
ax.set_xlabel("log(fitted HRs)")
ax.set_ylabel("√|Pearson residuals|")
ax.set_title("(c) Scale-location")

# (d) Deviance residuals vs log(fitted)
ax = axes3[1, 0]
ax.scatter(np.log(fitted_m0), deviance_m0, s=5, alpha=0.35, color="steelblue")
ax.axhline(0, color="black", lw=0.8)
ax.axhline(2,  color="orange", lw=0.9, ls="--")
ax.axhline(-2, color="orange", lw=0.9, ls="--")
ax.set_xlabel("log(fitted HRs)")
ax.set_ylabel("Deviance residuals")
ax.set_title("(d) Deviance residuals vs Fitted")

# (e) Actual vs predicted
ax = axes3[1, 1]
lim_val = max(df["home_run"].max(), fitted_m0.max()) + 2
ax.scatter(fitted_m0, df["home_run"], s=5, alpha=0.25, color="steelblue")
ax.plot([0, lim_val], [0, lim_val], "r--", lw=1, label="y = x")
ax.set_xlabel("Predicted HRs")
ax.set_ylabel("Actual HRs")
ax.set_title("(e) Actual vs Predicted")
ax.legend(fontsize=8)

# (f) Histogram of Pearson residuals
ax = axes3[1, 2]
ax.hist(pearson_m0, bins=40, color="steelblue", edgecolor="white", alpha=0.82)
ax.axvline(0, color="red", lw=1)
ax.set_xlabel("Pearson residuals")
ax.set_ylabel("Count")
ax.set_title("(f) Distribution of Pearson residuals")

plt.tight_layout()
plt.savefig("fig3_m0_diagnostics_v2.png", bbox_inches="tight")
plt.close()
print("  → fig3_m0_diagnostics_v2.png")

pct_out_2 = (np.abs(pearson_m0) > 2).mean() * 100
print(f"\n  M_0 Pearson residuals: mean={pearson_m0.mean():.4f}, "
      f"std={pearson_m0.std():.4f}")
print(f"  {pct_out_2:.1f}% outside ±2  (expect ~5% under correct Poisson model)")
print(f"  Residual deviance/df = {disp_m0:.3f}  "
      f"({'overdispersed' if disp_m0 > 1.5 else 'acceptable'})")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  STEPWISE MODEL SELECTION — AIC AND BIC  (Poisson)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("8. STEPWISE VARIABLE SELECTION (Poisson GLM)")
print("=" * 72)

# Full set of candidate predictors for the stepwise search.
# Replace barrel_batted_rate_lag with its log1p-transformed version;
# drop the untransformed column from the search.
ALL_CANDIDATES = (
    [c for c in LAG_LAG if c != "barrel_batted_rate_lag"]
    + ["log_barrel_lag", "player_age", "age2"]
)


def fit_poisson(df, variables, response="home_run", offset_col="log_pa"):
    """Fit a Poisson GLM and return the fitted model object."""
    formula = (f"{response} ~ {' + '.join(variables)}"
               if variables else f"{response} ~ 1")
    return smf.glm(
        formula, data=df,
        family=sm.families.Poisson(),
        offset=df[offset_col],
    ).fit(disp=0)


def stepwise_poisson(df, candidates, fixed_vars=None,
                     criterion="aic", verbose=True):
    """
    Greedy bidirectional stepwise selection for a Poisson GLM.

    At each iteration we compute the best possible single addition and
    the best possible single removal from the current model, then take
    whichever action most decreases the chosen information criterion.
    We stop when neither move improves the criterion.

    Parameters
    ----------
    candidates  : full list of predictors to search over
    fixed_vars  : predictors always kept in the model (not removed)
    criterion   : 'aic' or 'bic'
    """
    if fixed_vars is None:
        fixed_vars = []
    current = list(fixed_vars)

    def score(vars_list):
        m = fit_poisson(df, vars_list)
        if criterion == "aic":
            return m.aic
        else:
            # statsmodels GLM .bic uses a deviance formula that can be negative
            # and is not directly comparable to NB model .bic.
            # Use the standard BIC = -2*llf + k*log(n) instead.
            k = len(m.params)
            return -2.0 * m.llf + k * np.log(len(df))

    best = score(current)
    if verbose:
        label = current if current else ["intercept only"]
        print(f"  Start  {criterion.upper()} = {best:.2f}  |  vars: {label}")

    # Generous cap on iterations to prevent infinite loops
    for _ in range(len(candidates) * 2 + 10):
        options = {}

        # Try adding each variable not currently in the model
        for var in candidates:
            if var not in current:
                options[("add", var)] = score(current + [var])

        # Try removing each non-fixed variable currently in the model
        for var in current:
            if var not in fixed_vars:
                reduced = [v for v in current if v != var]
                options[("remove", var)] = score(reduced)

        if not options:
            break  # nothing left to try

        (action, var), new_score = min(options.items(), key=lambda kv: kv[1])

        if new_score >= best - 1e-4:
            break  # no meaningful improvement

        best = new_score
        if action == "add":
            current.append(var)
        else:
            current.remove(var)

        if verbose:
            arrow = "+" if action == "add" else "−"
            print(f"  {arrow} {var:<40}  {criterion.upper()} = {best:.2f}")

    return current, best


print("\n── AIC stepwise (Poisson) ──────────────────────────────────────────────")
aic_vars, aic_val = stepwise_poisson(df, ALL_CANDIDATES, criterion="aic")
m_pois_aic = fit_poisson(df, aic_vars)
bic_pois_aic = -2.0 * m_pois_aic.llf + len(m_pois_aic.params) * np.log(len(df))
print(f"\n  AIC model: AIC = {aic_val:.2f}, BIC = {bic_pois_aic:.2f}")
print(f"  {len(aic_vars)} predictors: {aic_vars}")

print("\n── BIC stepwise (Poisson) ──────────────────────────────────────────────")
bic_vars, bic_val = stepwise_poisson(df, ALL_CANDIDATES, criterion="bic")
m_pois_bic = fit_poisson(df, bic_vars)
bic_pois_bic = -2.0 * m_pois_bic.llf + len(m_pois_bic.params) * np.log(len(df))
print(f"\n  BIC model: AIC = {m_pois_bic.aic:.2f}, BIC = {bic_pois_bic:.2f}")
print(f"  {len(bic_vars)} predictors: {bic_vars}")

print("\n── AIC model summary ───────────────────────────────────────────────────")
print(m_pois_aic.summary())

print("\nFitted formula — M_AIC (Poisson):")
print("  log E[HR_t] = log(PA_t)")
for name, coef in m_pois_aic.params.items():
    print(f"    + ({coef:+.5f}) · {name}")

disp_aic_pois = m_pois_aic.deviance / m_pois_aic.df_resid
print(f"\n  M_AIC deviance/df = {disp_aic_pois:.3f}")

print("\n── BIC model summary ───────────────────────────────────────────────────")
print(m_pois_bic.summary())

print("\nFitted formula — M_BIC (Poisson):")
print("  log E[HR_t] = log(PA_t)")
for name, coef in m_pois_bic.params.items():
    print(f"    + ({coef:+.5f}) · {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LIKELIHOOD RATIO TESTS (LRT) — DROP-ONE TESTS ON M_AIC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("9. LIKELIHOOD RATIO TESTS (LRT) — DROP-ONE FROM M_AIC (POISSON)")
print("=" * 72)

print("""
For each predictor in M_AIC we test
  H₀ : the model without that variable fits equally well
  H₁ : the full M_AIC

  LRT = 2(ℓ_full − ℓ_reduced)  ~  χ²(1)  under H₀

A variable with p < 0.05 significantly improves the model and should
be kept; a non-significant variable could potentially be removed.
""")

print(f"  {'Variable':<42} {'LRT':>8} {'p-value':>12}  Sig")
print("  " + "─" * 68)

lrt_results = {}
for var in aic_vars:
    reduced_vars = [v for v in aic_vars if v != var]
    m_red = fit_poisson(df, reduced_vars)
    lrt_stat = 2.0 * (m_pois_aic.llf - m_red.llf)
    p_val = stats.chi2.sf(lrt_stat, df=1)
    lrt_results[var] = (lrt_stat, p_val)
    sig = ("***" if p_val < 0.001 else
           "**"  if p_val < 0.01  else
           "*"   if p_val < 0.05  else "")
    print(f"  {var:<42} {lrt_stat:>8.2f} {p_val:>12.4g}  {sig}")

print("\n  Significance codes: *** p<0.001  ** p<0.01  * p<0.05")

# Identify any variables that are not significant at 5 %
non_sig = [v for v, (_, p) in lrt_results.items() if p >= 0.05]
if non_sig:
    print(f"\n  Variables not significant at α=0.05: {non_sig}")
    print("  These could be removed from M_AIC; the BIC model may have "
          "already excluded them.")
else:
    print("\n  All variables in M_AIC are significant at α=0.05 by LRT.")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  NEGATIVE BINOMIAL REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("10. NEGATIVE BINOMIAL (NB2) REGRESSION")
print("=" * 72)

print("""
Given the overdispersion detected above, we now fit Negative Binomial
(NB2) models using the same variable sets from M_0 and M_AIC.

The NB2 model adds a free dispersion parameter α such that
  Var(Y) = μ + α · μ²
When α → 0, NB2 reduces to Poisson; α > 0 indicates extra-Poisson
variance (overdispersion).

We test Poisson vs NB2 using a boundary LRT:
  LRT = 2(ℓ_NB − ℓ_Poisson) ~ 0.5·χ²(0) + 0.5·χ²(1)  under H₀(Poisson)
  → p-value ≈ 0.5 · P(χ²(1) > LRT)
""")

def fit_nb(df, variables, response="home_run", offset_col="log_pa"):
    """Fit a Negative Binomial (NB2) GLM and return the model."""
    formula = (f"{response} ~ {' + '.join(variables)}"
               if variables else f"{response} ~ 1")
    return smf.negativebinomial(
        formula, data=df,
        offset=df[offset_col],
    ).fit(disp=False, method="bfgs", maxiter=200)


def nb_fitted_counts(model, df, offset_col="log_pa"):
    """
    Return expected HR counts from a fitted NB2 (smf.negativebinomial) model.

    smf.negativebinomial stores fittedvalues as the LINEAR PREDICTOR Xβ
    (without the offset), not the expected counts.  We must add the offset
    (log PA) and exponentiate to recover E[HR] = PA · exp(Xβ).
    """
    linear_predictor = model.fittedvalues.values
    if linear_predictor.min() < 0:
        # Confirmed: linear predictor scale (Xβ without offset)
        return np.exp(linear_predictor + df[offset_col].values)
    # If already positive (response scale), return as-is
    return linear_predictor


# Fit NB for M_0 and M_AIC variable sets
n_obs = len(df)   # used in BIC = -2*llf + k*log(n) throughout

print("  Fitting NB2 models …")
m_nb_0   = fit_nb(df, M0_VARS)
m_nb_aic = fit_nb(df, aic_vars)
m_nb_bic = fit_nb(df, bic_vars)

# Extract alpha (NB dispersion parameter) — it appears in params as 'alpha'
alpha_0   = m_nb_0.params.get("alpha", np.nan)
alpha_aic = m_nb_aic.params.get("alpha", np.nan)
alpha_bic = m_nb_bic.params.get("alpha", np.nan)

print(f"\n  M_0   NB2:  AIC = {m_nb_0.aic:.2f},  BIC = {-2*m_nb_0.llf + (len(m_nb_0.params))*np.log(n_obs):.2f},  α = {alpha_0:.4f}")
print(f"  M_AIC NB2:  AIC = {m_nb_aic.aic:.2f},  BIC = {-2*m_nb_aic.llf + (len(m_nb_aic.params))*np.log(n_obs):.2f},  α = {alpha_aic:.4f}")
print(f"  M_BIC NB2:  AIC = {m_nb_bic.aic:.2f},  BIC = {-2*m_nb_bic.llf + (len(m_nb_bic.params))*np.log(n_obs):.2f},  α = {alpha_bic:.4f}")

# LRT: Poisson vs NB2  (boundary test, one-sided p = 0.5 * chi2.sf)
def pois_vs_nb_lrt(m_pois, m_nb, label=""):
    lrt = 2.0 * (m_nb.llf - m_pois.llf)
    p   = 0.5 * stats.chi2.sf(lrt, df=1)   # one-sided boundary correction
    print(f"  {label:<20}  LRT = {lrt:.2f},  boundary p = {p:.4g}",
          "→ NB preferred" if p < 0.05 else "→ Poisson adequate")

print("\n  Poisson vs NB2 LRT (boundary-corrected):")
pois_vs_nb_lrt(m_pois_0,   m_nb_0,   "M_0")
pois_vs_nb_lrt(m_pois_aic, m_nb_aic, "M_AIC")
pois_vs_nb_lrt(m_pois_bic, m_nb_bic, "M_BIC")

# Print AIC-model NB summary
print("\n── M_AIC NB2 summary ───────────────────────────────────────────────────")
print(m_nb_aic.summary())

print("\nFitted formula — M_AIC (NB2):")
print("  log E[HR_t] = log(PA_t)")
nb_params = {k: v for k, v in m_nb_aic.params.items() if k != "alpha"}
for name, coef in nb_params.items():
    print(f"    + ({coef:+.5f}) · {name}")
print(f"  Dispersion parameter  α = {alpha_aic:.4f}  "
      f"[Var(Y|X) = μ + {alpha_aic:.4f}·μ²]")

# NegativeBinomialResults does not expose .deviance; use McFadden pseudo-R² instead.
# McFadden R² = 1 - (log-likelihood of fitted model) / (log-likelihood of null model)
pseudo_r2_nb_aic = 1.0 - m_nb_aic.llf / m_nb_aic.llnull
print(f"\n  M_AIC NB2 McFadden pseudo-R² = {pseudo_r2_nb_aic:.4f}")

# Note on multicollinearity: the x-stats (xba, xslg, xiso, xwoba, xwobacon, xbacon)
# are highly correlated.  Their coefficients can be large with opposite signs while
# jointly fitting well — a classic VIF problem.  The stepwise selected them because
# they collectively reduce AIC, but individual p-values may be inflated by collinearity.
print("\n  NOTE: Several x-stat predictors are highly correlated (xba, xslg, xiso,")
print("  xwoba, xwobacon, xbacon).  The large opposing coefficients suggest")
print("  multicollinearity.  Treat individual p-values for these terms with caution;"
      )
print("  the model's overall fit and outlier detection remain valid.")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  BRIER SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("11. BRIER SCORES")
print("=" * 72)

print("""
Adapting the Brier score to a count-response model
───────────────────────────────────────────────────
The Brier score requires a binary event and a predicted probability.
We define:

    Event  E_i = 1  if player i HIT MORE HRs than predicted
               = 0  otherwise
    i.e.  E_i = 𝟙[ HR_i > floor(μ̂_i) ]

Predicted probability (Poisson):
    p_i = P(HR > floor(μ̂_i) | λ = μ̂_i)
        = 1 − Poisson_CDF( floor(μ̂_i) ; μ̂_i )

Predicted probability (NB2):
    p_i = 1 − NB_CDF( floor(μ̂_i) ; r=1/α, p=1/(1+α·μ̂_i) )

Brier score = mean[ (p_i − E_i)² ]     (lower = better; 0.25 = random)
""")


def brier_poisson(fitted, actuals):
    """Brier score for 'exceeded predicted HRs' event, Poisson distribution."""
    threshold = np.floor(fitted).astype(int)
    p_exceed  = 1.0 - sp_poisson.cdf(threshold, fitted)
    y_exceed  = (actuals > threshold).astype(float)
    return float(np.mean((p_exceed - y_exceed) ** 2))


def brier_nb(fitted, actuals, alpha):
    """Brier score for 'exceeded predicted HRs' event, NB2 distribution."""
    threshold = np.floor(fitted).astype(int)
    r = 1.0 / alpha                          # NB shape parameter
    p = 1.0 / (1.0 + alpha * fitted)        # NB success probability
    p_exceed  = 1.0 - sp_nbinom.cdf(threshold, r, p)
    y_exceed  = (actuals > threshold).astype(float)
    return float(np.mean((p_exceed - y_exceed) ** 2))


# Null model: every player predicted at the overall HR rate (no covariates)
overall_rate = df["home_run"].sum() / df["pa"].sum()
null_fitted  = df["pa"] * overall_rate
bs_null = brier_poisson(null_fitted.values, df["home_run"].values)

y = df["home_run"].values
bs_pois_0   = brier_poisson(m_pois_0.fittedvalues.values,   y)
bs_pois_aic = brier_poisson(m_pois_aic.fittedvalues.values, y)
bs_pois_bic = brier_poisson(m_pois_bic.fittedvalues.values, y)
bs_nb_0     = brier_nb(nb_fitted_counts(m_nb_0,   df), y, alpha_0)
bs_nb_aic   = brier_nb(nb_fitted_counts(m_nb_aic, df), y, alpha_aic)
bs_nb_bic   = brier_nb(nb_fitted_counts(m_nb_bic, df), y, alpha_bic)

print(f"  {'Model':<25}  {'Brier':>8}  {'AIC':>10}  {'BIC':>10}  {'k':>4}  {'disp/df':>8}")
print("  " + "─" * 72)

n_obs = len(df)

# Standard BIC = -2*llf + k*log(n) for all models
def std_bic(model):
    k = len(model.params)
    return -2.0 * model.llf + k * np.log(n_obs)

null_aic     = m_null_pois.aic
null_bic_std = std_bic(m_null_pois)

# Poisson: deviance/df ratio measures residual overdispersion
disp_pois_0   = m_pois_0.deviance   / m_pois_0.df_resid
disp_pois_aic = m_pois_aic.deviance / m_pois_aic.df_resid
disp_pois_bic = m_pois_bic.deviance / m_pois_bic.df_resid
# NB2: McFadden pseudo-R² (no .deviance attribute on NegativeBinomialResults)
pseudo_r2_nb_0   = 1.0 - m_nb_0.llf   / m_nb_0.llnull
pseudo_r2_nb_bic = 1.0 - m_nb_bic.llf / m_nb_bic.llnull

rows = [
    ("Null (mean rate)",  bs_null,     null_aic,           null_bic_std,        1,               "disp/df", "—"),
    ("M_0  Poisson",      bs_pois_0,   m_pois_0.aic,       std_bic(m_pois_0),   len(M0_VARS),    "disp/df", f"{disp_pois_0:.3f}"),
    ("M_AIC Poisson",     bs_pois_aic, m_pois_aic.aic,     std_bic(m_pois_aic), len(aic_vars),   "disp/df", f"{disp_pois_aic:.3f}"),
    ("M_BIC Poisson",     bs_pois_bic, m_pois_bic.aic,     std_bic(m_pois_bic), len(bic_vars),   "disp/df", f"{disp_pois_bic:.3f}"),
    ("M_0  NB2",          bs_nb_0,     m_nb_0.aic,         std_bic(m_nb_0),     len(M0_VARS)+1,  "pR2",     f"{pseudo_r2_nb_0:.4f}"),
    ("M_AIC NB2",         bs_nb_aic,   m_nb_aic.aic,       std_bic(m_nb_aic),   len(aic_vars)+1, "pR2",     f"{pseudo_r2_nb_aic:.4f}"),
    ("M_BIC NB2",         bs_nb_bic,   m_nb_bic.aic,       std_bic(m_nb_bic),   len(bic_vars)+1, "pR2",     f"{pseudo_r2_nb_bic:.4f}"),
]

print("  NOTE: BIC uses the standard formula -2*llf + k*log(n) for all models.")
print(f"  {'Model':<25}  {'Brier':>8}  {'AIC':>10}  {'BIC':>10}  {'k':>4}  {'stat':>5}  {'value':>8}")
print("  " + "-" * 74)
for name, bs, aic_v, bic_v, k, stat_lbl, stat_val in rows:
    print(f"  {name:<25}  {bs:>8.5f}  {aic_v:>10.2f}  {bic_v:>10.2f}  "
          f"{str(k):>4}  {stat_lbl:>5}  {stat_val:>8}")

print("\n  Brier: lower = better calibration (0.25 = random coin flip).")
print("  AIC/BIC: lower = better penalised fit.")
print("  disp/df (Poisson): ~1.0 ideal; >>1 = overdispersion.")
print("  pR2 (NB2): McFadden pseudo-R² = 1 - llf/llnull; higher = better.")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  OUTLIER IDENTIFICATION  (using best NB model = M_AIC NB2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("12. OUTLIER IDENTIFICATION — M_AIC NB2 MODEL")
print("=" * 72)

print("""
We use the M_AIC NB2 model as our preferred specification (NB2
accounts for overdispersion; AIC-selected variables are the most
informative).

Outliers are flagged by large STANDARDISED Pearson residuals:

    r_P = (y − μ̂) / sqrt(Var̂(Y))

For NB2: Var̂(Y_i) = μ̂_i + α · μ̂_i²
""")

mu_nb  = nb_fitted_counts(m_nb_aic, df)
# Guard against near-zero fitted values causing division by zero
var_nb = np.maximum(mu_nb + alpha_aic * mu_nb ** 2, 1e-8)
pearson_nb = (df["home_run"].values - mu_nb) / np.sqrt(var_nb)

# Report any non-finite residuals (symptoms of near-zero fitted values)
n_nan = np.sum(~np.isfinite(pearson_nb))
if n_nan > 0:
    print(f"  Warning: {n_nan} non-finite NB Pearson residuals; these rows are excluded.")
    finite_mask = np.isfinite(pearson_nb)
else:
    finite_mask = np.ones(len(pearson_nb), dtype=bool)

print(f"  NB Pearson residuals (n={finite_mask.sum()}): "
      f"mean={pearson_nb[finite_mask].mean():.3f}, "
      f"std={pearson_nb[finite_mask].std():.3f}, "
      f"min={pearson_nb[finite_mask].min():.2f}, "
      f"max={pearson_nb[finite_mask].max():.2f}")

df["fitted_nb"]      = mu_nb
df["pearson_nb"]     = pearson_nb
df["abs_pearson_nb"] = np.abs(pearson_nb)

# Use 2.5 as the outlier threshold: NB residuals tend to be smaller than
# Poisson residuals because NB variance > Poisson variance.  At n~3800,
# ~1% of observations (38) are expected outside ±2.5 under the correct model.
THRESHOLD = 2.5
outliers = df[df["abs_pearson_nb"] > THRESHOLD].sort_values(
    "pearson_nb", ascending=False
).copy()

print(f"Player-seasons with |NB Pearson residual| > {THRESHOLD:.1f}:\n")
header = (f"  {'Player':<26} {'Year':>5} {'Age':>4} {'PA':>5} "
          f"{'Actual HR':>10} {'Predicted':>10} {'NB Pearson R':>13}")
print(header)
print("  " + "─" * 76)
for _, row in outliers.iterrows():
    direction = "↑ HIGH" if row["pearson_nb"] > 0 else "↓ LOW"
    print(f"  {row['player_name']:<26} {int(row['year']):>5} "
          f"{int(row['player_age']):>4} {int(row['pa']):>5} "
          f"{int(row['home_run']):>10} {row['fitted_nb']:>10.1f} "
          f"{row['pearson_nb']:>+10.2f} {direction}")

# ── Figure: actual vs predicted with outliers labelled ─────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))
fig4.suptitle("Outlier identification — M_AIC NB2 model", fontsize=11)

# Panel A: actual vs predicted
ax = axes4[0]
normal  = df[df["abs_pearson_nb"] <= THRESHOLD]
ax.scatter(normal["fitted_nb"],   normal["home_run"],
           s=6, alpha=0.25, color="steelblue", label="Normal")
ax.scatter(outliers["fitted_nb"], outliers["home_run"],
           s=50, alpha=0.85, color="red", marker="^", label=f"|r| > {THRESHOLD:.0f}")

# Label each outlier
for _, row in outliers.iterrows():
    name_short = row["player_name"].split(",")[0]
    yr_short   = str(int(row["year"]))[-2:]
    ax.annotate(
        f"{name_short} '{yr_short}",
        xy=(row["fitted_nb"], row["home_run"]),
        xytext=(5, 2), textcoords="offset points",
        fontsize=6.5, color="darkred",
    )

lim_val = max(df["home_run"].max(), df["fitted_nb"].max()) + 3
ax.plot([0, lim_val], [0, lim_val], "k--", lw=0.9, label="y = x")
ax.set_xlabel("Predicted HRs (M_AIC NB2)")
ax.set_ylabel("Actual HRs")
ax.set_title("Actual vs Predicted")
ax.legend(fontsize=8)

# Panel B: NB Pearson residuals vs log(fitted)
ax = axes4[1]
ax.scatter(np.log(mu_nb), pearson_nb, s=6, alpha=0.25, color="steelblue")
for lvl, col in [(2, "orange"), (3, "red")]:
    ax.axhline( lvl, color=col, lw=0.9, ls="--", label=f"±{lvl}")
    ax.axhline(-lvl, color=col, lw=0.9, ls="--")
ax.axhline(0, color="black", lw=0.7)

for _, row in outliers.iterrows():
    name_short = row["player_name"].split(",")[0]
    yr_short   = str(int(row["year"]))[-2:]
    ax.annotate(
        f"{name_short} '{yr_short}",
        xy=(np.log(row["fitted_nb"]), row["pearson_nb"]),
        xytext=(5, 2), textcoords="offset points",
        fontsize=6.5, color="darkred",
    )

ax.set_xlabel("log(Predicted HRs)")
ax.set_ylabel("NB Pearson residuals")
ax.set_title("Residuals vs Fitted  (±2, ±3 bands)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig4_outliers_v2.png", bbox_inches="tight")
plt.close()
print(f"\n  → fig4_outliers_v2.png")

# ── Final NB diagnostic plots ───────────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4.5))
fig5.suptitle("M_AIC NB2 — final diagnostics", fontsize=11)

ax = axes5[0]
finite_resids_qq = pearson_nb[np.isfinite(pearson_nb)]
(osm, osr), (slope, inter, _) = stats.probplot(finite_resids_qq, dist="norm")
ax.plot(osm, osr, "o", ms=3, alpha=0.35, color="steelblue")
ax.plot(osm, slope * np.array(osm) + inter, color="red", lw=1.2)
ax.set_title("Q-Q plot — NB Pearson residuals")
ax.set_xlabel("Theoretical quantiles  N(0,1)")
ax.set_ylabel("Sample quantiles")

ax = axes5[1]
finite_resids = pearson_nb[np.isfinite(pearson_nb)]
ax.hist(finite_resids, bins=40, color="steelblue", edgecolor="white", alpha=0.82)
ax.axvline(0, color="red", lw=1)
ax.set_xlabel("NB Pearson residuals")
ax.set_ylabel("Count")
ax.set_title("Distribution of NB residuals")

ax = axes5[2]
finite_mu   = mu_nb[np.isfinite(pearson_nb)]
finite_rabs = np.abs(pearson_nb[np.isfinite(pearson_nb)])
ax.scatter(np.log(finite_mu), np.sqrt(finite_rabs),
           s=5, alpha=0.3, color="steelblue")
ax.set_xlabel("log(Fitted HRs)")
ax.set_ylabel("√|NB Pearson residuals|")
ax.set_title("Scale-location (M_AIC NB2)")

plt.tight_layout()
plt.savefig("fig5_nb_diagnostics_v2.png", bbox_inches="tight")
plt.close()
print("  → fig5_nb_diagnostics_v2.png")

pct_out_nb = (np.abs(finite_resids_qq) > 2).mean() * 100
print(f"\n  M_AIC NB2: {pct_out_nb:.1f}% of NB Pearson residuals outside ±2")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY — ALL MODELS")
print("=" * 72)

print(f"\n  {'Model':<25}  {'AIC':>10}  {'BIC*':>10}  {'Brier':>8}  {'fit stat':>10}  {'k':>4}")
print("  * BIC = -2·llf + k·log(n)  (standard formula)")
print("  " + "-" * 74)
# disp/df for Poisson models; McFadden pR2 for NB2 models
summary_rows = [
    ("Null Poisson",  null_aic,         null_bic_std,        bs_null,     "disp/df=—",                     1),
    ("M_0  Poisson",  m_pois_0.aic,     std_bic(m_pois_0),   bs_pois_0,   f"disp/df={disp_pois_0:.2f}",    len(M0_VARS)),
    ("M_AIC Poisson", m_pois_aic.aic,   std_bic(m_pois_aic), bs_pois_aic, f"disp/df={disp_pois_aic:.2f}",  len(aic_vars)),
    ("M_BIC Poisson", m_pois_bic.aic,   std_bic(m_pois_bic), bs_pois_bic, f"disp/df={disp_pois_bic:.2f}",  len(bic_vars)),
    ("M_0  NB2",      m_nb_0.aic,       std_bic(m_nb_0),     bs_nb_0,     f"pR2={pseudo_r2_nb_0:.3f}",     len(M0_VARS)+1),
    ("M_AIC NB2",     m_nb_aic.aic,     std_bic(m_nb_aic),   bs_nb_aic,   f"pR2={pseudo_r2_nb_aic:.3f}",   len(aic_vars)+1),
    ("M_BIC NB2",     m_nb_bic.aic,     std_bic(m_nb_bic),   bs_nb_bic,   f"pR2={pseudo_r2_nb_bic:.3f}",   len(bic_vars)+1),
]
for name, aic_v, bic_v, bs, fit_stat, k in summary_rows:
    bs_str = f"{bs:.5f}" if isinstance(bs, float) else str(bs)
    print(f"  {name:<25}  {aic_v:>10.2f}  {bic_v:>10.2f}  "
          f"{bs_str:>8}  {fit_stat:>10}  {str(k):>4}")

print("""
Interpretation notes
────────────────────
  * AIC/BIC both penalise log-likelihood for model complexity — lower is better.
  * Brier score measures calibration of the probability forecast for the
    event "player exceeded expected HRs" — lower is better; 0.25 = random.
  * disp/df = residual deviance / df_residual; should be ≈ 1 for Poisson,
    less important for NB2 (which has its own dispersion parameter).
  * The preferred model is M_AIC NB2 (or M_BIC NB2 if parsimony is valued).

Output figures
──────────────
  fig1_predictor_distributions_v2.png  – histograms of all lagged predictors
  fig2_predictor_vs_hr_rate_v2.png     – scatter plots vs log(HR rate)
  fig3_m0_diagnostics_v2.png           – 6-panel diagnostics for M_0 (Poisson)
  fig4_outliers_v2.png                 – actual vs predicted + residual plot
  fig5_nb_diagnostics_v2.png           – Q-Q and residual plots for M_AIC NB2
""")
