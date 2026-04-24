"""
Main runner: end-to-end evaluation of a trailing stop on multiple strategies.

Tests three stop rules head-to-head against a no-stop baseline:
  - NoStop         : no risk rule applied
  - HardStop       : full exit at $2m drawdown from HWM, no re-entry
  - OneReduction   : cut to 50% at $1m, full exit at $2m, no re-entry
  - TwoReductions  : cut to 70% at $400k, to 40% at $1.1m, full exit at $2m,
                     step back up after $300k recovery from trough

Each strategy runs with its own capital allocation. A combined-sleeve
portfolio (sum of per-strategy equity curves) is also evaluated, which
captures the natural diversification across strategies.

Produces:
  - core diagnostics (percentiles, CVaR, drawdown dynamics, paired, conditional)
  - institutional one-pager (summary table + 4-panel plot per strategy + combined)
  - robustness checks (sensitivity to bootstrap L and rule parameters)

Replace the synthetic data in the SETUP section with your real strategy returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import (
    # simulation & engine
    generate_scenarios, NoStop, TrailingStopRule, run_backtest,
    # core analysis
    percentile_table, cvar_table,
    drawdown_summary, conditional_comparison,
    paired_comparison, bootstrap_ci,
    plot_distribution_overlay, combine_sleeves,
    # institutional
    institutional_summary, rolling_return_stats, dd_threshold_probabilities,
    stop_activity,
    plot_equity_fan, plot_drawdown_fan, plot_return_vs_dd_scatter,
    plot_did_stop_help,
    # sensitivity
    sensitivity_to_L, sensitivity_to_rule_params,
)


# =============================================================================
# SETUP — replace with your real data
# =============================================================================
dates = pd.bdate_range("2006-01-01", "2026-01-01")
n = len(dates)
rng = np.random.default_rng(0)

def ar1(n, mu, sigma, phi):
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = mu + phi * (r[i - 1] - mu) + eps[i]
    return r

historical_returns = {
    'momentum':    pd.Series(ar1(n, 0.0006, 0.014,  0.10), index=dates),
    'mean_revert': pd.Series(ar1(n, 0.0004, 0.009, -0.05), index=dates),
    'vol_carry':   pd.Series(ar1(n, 0.0008, 0.020,  0.08), index=dates),
}

# Per-strategy capital allocations. Stop levels below are absolute dollar
# amounts, so the rule is tighter in DD% at higher capital.
CAPITALS = {
    'momentum':    10_000_000.0,
    'mean_revert':  5_000_000.0,
    'vol_carry':   20_000_000.0,
}

# Stop rule parameters.
TWO_RED_LEVELS  = [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.00)]
ONE_RED_LEVELS  = [(1_000_000, 0.50), (2_000_000, 0.00)]
HARD_LEVELS     = [(2_000_000, 0.00)]
REENTRY_TWO_RED = 300_000
REENTRY_ONE_RED = 0          # keep simple for cleaner cross-rule comparison
REENTRY_HARD    = 0

# DD levels reported in drawdown_summary. Union of all rules' trigger points.
TRIGGER_DOLLARS = [400_000, 1_000_000, 1_100_000, 2_000_000]


# Factories (fresh rule instance per call — rules carry internal state).
def make_hard():
    return TrailingStopRule(levels=HARD_LEVELS, reentry_recovery=REENTRY_HARD,
                            label="HardStop-2m")

def make_one_red():
    return TrailingStopRule(levels=ONE_RED_LEVELS, reentry_recovery=REENTRY_ONE_RED,
                            label="OneRed-1m50%")

def make_two_red():
    return TrailingStopRule(levels=TWO_RED_LEVELS, reentry_recovery=REENTRY_TWO_RED,
                            label="TwoRed-400k70-1.1m40")

RULES = [
    ('baseline', lambda: NoStop()),
    ('hard',     make_hard),
    ('one_red',  make_one_red),
    ('two_red',  make_two_red),
]
# Short key -> label stored on the BacktestResult (used by combine_sleeves).
RULE_LABEL = {'baseline': 'NoStop',       'hard': 'HardStop-2m',
              'one_red':  'OneRed-1m50%', 'two_red': 'TwoRed-400k70-1.1m40'}


# =============================================================================
# STEP 1: generate scenarios ONCE, reuse for all strategies and rules
# =============================================================================
# Memory note: 3 strategies x 4 rules x n_paths x 1261 days x 8 bytes of
# equity curves is ~1.2 GB at n_paths=10000. 5000 paths gives solid tail
# estimates (50-100 obs at the 1% tail) with half the memory footprint.
# Bump back to 10000 if memory allows and you want tighter CIs.
N_PATHS = 5000
print(f"Generating {N_PATHS:,} scenarios...")
scenarios = generate_scenarios(
    historical_returns=historical_returns,
    n_paths=N_PATHS, path_length=1260, L_mean=None, seed=42,
)
print(f"  Politis-White L per strategy: {scenarios['L_per_strategy']}")
print(f"  Block length used: L_mean = {scenarios['L_mean']:.1f}")
print(f"  Per-strategy capital: {CAPITALS}")

# Run every (strategy × rule) combination with per-strategy capital lookup.
results = {}
for strat, paths in scenarios['paths'].items():
    cap = CAPITALS[strat]
    for rule_key, factory in RULES:
        results[(strat, rule_key)] = run_backtest(paths, factory(), strat, cap)

# Build combined-sleeve (multi-strategy portfolio) result per rule.
# Each sleeve keeps its own per-strategy capital and its own rule-independent
# dynamics; we just sum the equity curves across strategies.
for rule_key, _ in RULES:
    combined = combine_sleeves(
        results=results,
        strategies=list(historical_returns.keys()),
        rule_label=rule_key,        # we keyed `results` by short rule_key
        capitals=CAPITALS,
        combined_name='combined',
    )
    # combine_sleeves copies rule_label into the new result's rule_name;
    # overwrite with the pretty label for nicer table output.
    combined.rule_name = RULE_LABEL[rule_key]
    results[('combined', rule_key)] = combined

all_results = list(results.values())
strategies_for_iter = list(historical_returns.keys()) + ['combined']


# =============================================================================
# STEP 2: institutional summary + rolling / DD-threshold / activity tables
# =============================================================================
print("\n" + "=" * 100)
print("INSTITUTIONAL SUMMARY (the numbers an allocator actually asks for)")
print("=" * 100)
summary = institutional_summary(all_results, dd_thresholds=(0.10, 0.15, 0.20, 0.30))
with pd.option_context('display.max_columns', None, 'display.width', 220,
                       'display.float_format', '{:.3f}'.format):
    print(summary.to_string(index=False))
summary.to_csv('institutional_summary.csv', index=False)

roll_1y = pd.DataFrame([rolling_return_stats(r, 252) for r in all_results])
dd_probs = pd.DataFrame([
    dd_threshold_probabilities(r, (0.05, 0.10, 0.15, 0.20, 0.30, 0.50))
    for r in all_results
])
activity = pd.DataFrame([stop_activity(r) for r in all_results])

print("\n--- Rolling 1yr return distribution ---")
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(roll_1y.to_string(index=False))

print("\n--- Drawdown breach probabilities ---")
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(dd_probs.to_string(index=False))

print("\n--- Stop activity (how often the rule actually fires) ---")
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(activity.to_string(index=False))


# =============================================================================
# STEP 3: core diagnostics
# =============================================================================
print("\n" + "=" * 100)
print("CORE DIAGNOSTICS")
print("=" * 100)

pct_table = percentile_table(all_results)
cvar_t = cvar_table(all_results, alphas=(0.01, 0.05, 0.10))
print("\n--- Percentile table (terminal return and max DD) ---")
with pd.option_context('display.max_columns', None, 'display.width', 240,
                       'display.float_format', '{:.3f}'.format):
    print(pct_table.to_string(index=False))
print("\n--- CVaR table ---")
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(cvar_t.to_string(index=False))

print("\n--- Drawdown dynamics (hit rates, time underwater, recovery) ---")
dd_rows = pd.DataFrame([drawdown_summary(r, TRIGGER_DOLLARS) for r in all_results])
with pd.option_context('display.max_columns', None, 'display.width', 240,
                       'display.float_format', '{:.3f}'.format):
    print(dd_rows.to_string(index=False))

# --- Paired comparisons
print("\n--- Paired comparisons and bootstrap CIs ---")
paired_rows, ci_rows = [], []
# (a) Each treatment vs NoStop baseline (is any stop worth using?)
for strat in strategies_for_iter:
    baseline = results[(strat, 'baseline')]
    for treated_key in ('hard', 'one_red', 'two_red'):
        treated = results[(strat, treated_key)]
        paired_rows.append(paired_comparison(treated, baseline))
        ci = bootstrap_ci(treated, baseline, 'total_returns', n_resamples=2000)
        ci['strategy'] = strat
        ci['comparison'] = f'{RULE_LABEL[treated_key]} vs NoStop'
        ci_rows.append(ci)

# (b) Pairwise across treatments (does tiering improve on a simpler rule?)
for strat in strategies_for_iter:
    for treated_key, anchor_key in [('one_red', 'hard'),
                                     ('two_red', 'hard'),
                                     ('two_red', 'one_red')]:
        treated = results[(strat, treated_key)]
        anchor  = results[(strat, anchor_key)]
        paired_rows.append(paired_comparison(treated, anchor))
        ci = bootstrap_ci(treated, anchor, 'total_returns', n_resamples=2000)
        ci['strategy'] = strat
        ci['comparison'] = f'{RULE_LABEL[treated_key]} vs {RULE_LABEL[anchor_key]}'
        ci_rows.append(ci)

with pd.option_context('display.float_format', '{:.4f}'.format,
                       'display.max_columns', None, 'display.width', 220):
    print(pd.DataFrame(paired_rows).to_string(index=False))
    print("\nBootstrap 95% CI on mean-return difference:")
    print(pd.DataFrame(ci_rows).to_string(index=False))

print("\n--- Conditional comparison (momentum, TwoRed vs NoStop, bucketed by worst_30d) ---")
cond = conditional_comparison(
    results[('momentum', 'two_red')], results[('momentum', 'baseline')],
    bucket_by='worst_30d', n_buckets=5,
)
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.max_columns', None, 'display.width', 200):
    print(cond.to_string(index=False))
print("Does the stop help most in bucket 0 (worst-tail paths)?")


# =============================================================================
# STEP 4: plots
# =============================================================================
rule_keys_for_plots = ['baseline', 'hard', 'one_red', 'two_red']

# Distribution overlays for all strategies × all rules.
fig, axes = plt.subplots(2, len(strategies_for_iter),
                         figsize=(5 * len(strategies_for_iter), 9))
for j, strat in enumerate(strategies_for_iter):
    plot_distribution_overlay(
        [results[(strat, k)] for k in rule_keys_for_plots],
        metric='total_returns', ax=axes[0, j],
        title=f'{strat}: terminal return')
    plot_distribution_overlay(
        [results[(strat, k)] for k in rule_keys_for_plots],
        metric='max_drawdown_pct', ax=axes[1, j],
        title=f'{strat}: max DD %')
plt.tight_layout()
plt.savefig('distribution_overlays.png', dpi=120)
plt.close()
print("\nSaved distribution_overlays.png")

# One-pager per strategy. Top row: equity fan + DD fan for all 4 rules.
# Bottom-left: return-vs-DD scatter for all 4. Bottom-right: did-it-help
# histogram comparing TwoRed to HardStop (the key "is tiering worth it" view).
for strat in strategies_for_iter:
    b    = results[(strat, 'baseline')]
    hard = results[(strat, 'hard')]
    oneR = results[(strat, 'one_red')]
    twoR = results[(strat, 'two_red')]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_equity_fan([b, hard, oneR, twoR], ax=axes[0, 0],
                    title=f'{strat}: equity fan (all rules)')
    plot_drawdown_fan([b, hard, oneR, twoR], ax=axes[0, 1],
                      title=f'{strat}: drawdown fan (all rules)')
    plot_return_vs_dd_scatter([b, hard, oneR, twoR], ax=axes[1, 0],
                              title=f'{strat}: return vs max DD')
    plot_did_stop_help(twoR, hard, ax=axes[1, 1],
                       title=f'{strat}: TwoRed vs HardStop per-path delta')
    plt.tight_layout()
    plt.savefig(f'onepager_{strat}.png', dpi=130)
    plt.close()
    print(f"Saved onepager_{strat}.png")


# =============================================================================
# STEP 5: robustness sensitivity sweeps
# =============================================================================
print("\n" + "=" * 100)
print("SENSITIVITY TO BOOTSTRAP BLOCK LENGTH L")
print("=" * 100)
print("Re-running each rule at L = 10, 30, 60, 120, 250 with per-strategy "
      "capital...")
# sensitivity_to_L now accepts a dict for initial_capital, so we can pass
# all strategies in a single call and let the module do the per-strategy
# lookup. One call per treatment rule, concat the results.
sens_pieces = []
for rule_key in ('hard', 'one_red', 'two_red'):
    factory = dict(RULES)[rule_key]
    df = sensitivity_to_L(
        historical_returns=historical_returns,
        rule_factory=factory,
        L_values=[10, 30, 60, 120, 250],
        n_paths=3000, path_length=1260,
        initial_capital=CAPITALS,           # dict -> per-strategy capital
        seed=42,
    )
    sens_pieces.append(df)
sens_L = pd.concat(sens_pieces, ignore_index=True)
# Drop duplicated NoStop rows (one per factory call).
sens_L = sens_L.drop_duplicates(subset=['L_mean', 'strategy', 'rule'])

pivot_tr = sens_L.pivot_table(index='L_mean', columns=['strategy', 'rule'],
                              values='mean_total_return')
with pd.option_context('display.float_format', '{:.4f}'.format,
                       'display.max_columns', None, 'display.width', 220):
    print("\nMean total return by L:")
    print(pivot_tr.to_string())

print("\n" + "=" * 100)
print("SENSITIVITY TO RULE PARAMETERS (varies the TwoRed structure only)")
print("=" * 100)
# Rule-parameter sensitivity tests whether the rule is robust to small changes
# in threshold levels — a question about overfitting, not capital calibration.
# We use a single representative capital (momentum's $10m) for simplicity.
# If you want per-strategy rule sensitivity, wrap this loop similarly to above.
level_variants = [
    [(300_000, 0.70), (825_000, 0.40),   (1_500_000, 0.0)],  # tight (-25%)
    [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.0)],  # base
    [(500_000, 0.70), (1_375_000, 0.40), (2_500_000, 0.0)],  # loose (+25%)
]
sens_p = sensitivity_to_rule_params(
    scenario_paths=scenarios['paths'],
    level_variants=level_variants,
    reentry_variants=[0, 150_000, 300_000, 500_000],
    initial_capital=CAPITALS['momentum'],
)
pivot_p = sens_p.pivot_table(
    index=['variant_idx', 'reentry_recovery'],
    columns='strategy', values='mean_total_return',
)
with pd.option_context('display.float_format', '{:.4f}'.format):
    print("Mean total return by (variant, reentry) per strategy:")
    print(pivot_p.to_string())
print("\nVariant 0 = tight (-25%), 1 = base, 2 = loose (+25%)")
print("Smooth gradients across variants = robust. Sharp differences = overfit.")

sens_L.to_csv('sensitivity_L.csv', index=False)
sens_p.to_csv('sensitivity_params.csv', index=False)
activity.to_csv('stop_activity.csv', index=False)
print("\nAll CSVs saved in the current working directory.")
