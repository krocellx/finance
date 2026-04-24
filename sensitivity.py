"""
Sensitivity / robustness sweeps.

Two main questions:
  1. Are conclusions stable as the bootstrap block length L varies?
     -> sensitivity_to_L
  2. Are conclusions stable as the stop rule parameters vary?
     -> sensitivity_to_rule_params

Both produce DataFrames you can plot or pivot to see whether the stop's
advantage is robust or an artifact of specific parameter choices.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Callable
from .simulation import generate_scenarios
from .stop_rules import StopRule, TrailingStopRule, NoStop
from .engine import run_backtest, BacktestResult


def sensitivity_to_L(
    historical_returns: dict,
    rule_factory: Callable[[], StopRule],
    L_values: Sequence[float],
    baseline_factory: Callable[[], StopRule] = NoStop,
    n_paths: int = 10_000,
    path_length: int = 1260,
    initial_capital: float = 10_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Re-run the full evaluation at multiple bootstrap block lengths.

    For each L, builds a fresh scenario set and backtests every strategy
    against both the treated rule and the baseline. Returns a long DataFrame
    with one row per (L, strategy, rule) combination.

    Rules are passed as FACTORIES (callables returning a fresh rule instance)
    because rules carry state and must be re-created for each backtest.

    Use this to check whether the stop's advantage is stable across L. If
    the advantage flips sign as L varies, the rule is exploiting an artifact
    of the assumed persistence structure rather than a real phenomenon.
    """
    rows = []
    for L in L_values:
        scenarios = generate_scenarios(
            historical_returns=historical_returns,
            n_paths=n_paths,
            path_length=path_length,
            L_mean=L,
            seed=seed,
        )
        for strat_name, paths in scenarios['paths'].items():
            for factory in (baseline_factory, rule_factory):
                rule = factory()
                res = run_backtest(paths, rule, strat_name, initial_capital)
                s = res.summary()
                s['L_mean'] = L
                rows.append(s)
    return pd.DataFrame(rows)


def sensitivity_to_rule_params(
    scenario_paths: dict,
    level_variants: Sequence[tuple],
    reentry_variants: Sequence[float],
    baseline: BacktestResult = None,
    initial_capital: float = 10_000_000.0,
) -> pd.DataFrame:
    """
    Sweep trailing-stop parameters against a fixed scenario set.

    Parameters
    ----------
    scenario_paths : dict[str, ndarray]
        Output of generate_scenarios()['paths']. Same scenarios used for all
        variants, so comparisons are apples-to-apples.
    level_variants : list of level-tuple-lists
        Each element is a list of (trigger_dd, size) tuples defining one rule.
        Example:
            [
                [(300_000, 0.70), (900_000, 0.40), (1_700_000, 0.0)],   # tight
                [(400_000, 0.70), (1_100_000, 0.40), (2_000_000, 0.0)], # base
                [(500_000, 0.70), (1_300_000, 0.40), (2_300_000, 0.0)], # loose
            ]
    reentry_variants : list of float
        Re-entry recovery amounts to try. Example: [0, 200_000, 300_000, 500_000].

    Returns
    -------
    DataFrame with one row per (strategy, level_variant, reentry_variant).

    Healthy rules show SMOOTH performance gradients as parameters vary. If
    base params look great but ±25% variants look terrible, you've overfit
    to arbitrary thresholds.
    """
    rows = []
    for i, levels in enumerate(level_variants):
        for reentry in reentry_variants:
            for strat_name, paths in scenario_paths.items():
                rule = TrailingStopRule(
                    levels=list(levels),
                    reentry_recovery=reentry,
                    label=f'variant{i}_re{int(reentry/1000)}k',
                )
                res = run_backtest(paths, rule, strat_name, initial_capital)
                s = res.summary()
                s['variant_idx'] = i
                s['levels'] = str(levels)
                s['reentry_recovery'] = reentry
                rows.append(s)
    return pd.DataFrame(rows)


def sensitivity_to_capital(
    scenario_paths: dict,
    rule_factory: Callable[[], StopRule],
    capital_values: Sequence[float],
    baseline_factory: Callable[[], StopRule] = NoStop,
) -> pd.DataFrame:
    """
    Check how rule behavior changes with starting equity.

    Matters because absolute-dollar thresholds ($400k/$1.1m/$2m) behave very
    differently at $5m vs $20m starting capital. If the rule's relative
    advantage collapses at higher capital, the thresholds are sized for a
    specific account size and won't scale.
    """
    rows = []
    for cap in capital_values:
        for strat_name, paths in scenario_paths.items():
            for factory in (baseline_factory, rule_factory):
                rule = factory()
                res = run_backtest(paths, rule, strat_name, cap)
                s = res.summary()
                s['initial_capital'] = cap
                rows.append(s)
    return pd.DataFrame(rows)
