"""
Strategy settings: the fixed set of knobs the strategy review agent
(strategy_review.py, Phase 5) is allowed to propose changing, each with
a hard limit enforced here in code — not merely requested of the model.
A proposal that violates a limit is clipped/filtered to the nearest
valid value, never rejected outright with no portfolio change, so a
partially-reasonable proposal still does something safe.

This module only knows about settings and validation — no file I/O
beyond load/save (deliberately simple: one JSON file), no network calls,
no Claude API. rebalance.py reads these settings to build the
environment and the final recommendation; strategy_review.py reads and
proposes changes to them.
"""
import json

EQUITY_SHARE_RANGE = (0.50, 0.85)  # bonds between 15% and 50% of the portfolio
MAX_TILT = 0.05  # a strategic per-ticker tilt is at most ±5%
RISK_LEVELS = ("conservative", "balanced", "aggressive")
DEFAULT_RISK_LEVEL = "balanced"

SETTINGS_PATH = "strategy_settings.json"

DEFAULT_SETTINGS = {
    "equity_share": 0.75,  # matches the Phase 0 hardcoded BOND_FLOOR of 0.25
    "per_ticker_tilts": {},
    "excluded_tickers": [],
    "risk_level": DEFAULT_RISK_LEVEL,
}


def load_settings(path: str = SETTINGS_PATH) -> dict:
    """Load strategy settings, falling back to DEFAULT_SETTINGS if the
    file doesn't exist yet (e.g. before the first strategy review has
    ever run)."""
    try:
        with open(path) as f:
            settings = json.load(f)
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)

    # Merge onto the defaults rather than trusting the file to have
    # every key — a hand-edited or older-format file shouldn't crash
    # the pipeline for a missing field.
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings)
    return merged


def save_settings(settings: dict, path: str = SETTINGS_PATH):
    with open(path, "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)


def max_excluded_tickers(n_equity_tickers: int) -> int:
    """
    At most this many equities may be excluded at once — enough remain
    that apply_exclusions (portfolio_lib.py) never has to collapse the
    entire equity sleeve to zero from a single strategy review decision.
    Requires at least half of the universe to remain investable.
    """
    return n_equity_tickers // 2


def validate_and_clip(proposed: dict, current: dict, equity_tickers) -> tuple:
    """
    Enforce every hard limit on a proposed settings dict, falling back
    to `current`'s value for any field so malformed that clipping
    doesn't make sense (e.g. an unparseable risk_level), rather than a
    fixed default — so one bad field doesn't silently revert every
    other setting on an otherwise-reasonable proposal.

    Returns (validated_settings: dict, corrections: list[str]) — the
    corrections list is empty if the proposal needed no changes at all,
    and is for logging/transparency, not for raising or blocking
    anything: every proposal always produces *some* valid settings dict.
    """
    validated = {}
    corrections = []

    # --- equity_share ---
    try:
        equity_share = float(proposed.get("equity_share", current["equity_share"]))
    except (TypeError, ValueError):
        equity_share = current["equity_share"]
        corrections.append(f"equity_share was not a number; kept current value {equity_share}")

    low, high = EQUITY_SHARE_RANGE
    if not (low <= equity_share <= high):
        clipped = min(max(equity_share, low), high)
        corrections.append(f"equity_share {equity_share} outside allowed range [{low}, {high}]; clipped to {clipped}")
        equity_share = clipped
    validated["equity_share"] = equity_share

    # --- per_ticker_tilts ---
    raw_tilts = proposed.get("per_ticker_tilts", {}) or {}
    tilts = {}
    if not isinstance(raw_tilts, dict):
        corrections.append("per_ticker_tilts was not a mapping; ignored entirely")
        raw_tilts = {}
    for ticker, tilt in raw_tilts.items():
        if ticker not in equity_tickers:
            corrections.append(f"tilt for unknown ticker {ticker!r} ignored")
            continue
        try:
            tilt = float(tilt)
        except (TypeError, ValueError):
            corrections.append(f"tilt for {ticker} was not a number; ignored")
            continue
        clipped = min(max(tilt, -MAX_TILT), MAX_TILT)
        if clipped != tilt:
            corrections.append(f"tilt for {ticker} ({tilt}) outside allowed range "
                                f"[-{MAX_TILT}, {MAX_TILT}]; clipped to {clipped}")
        if clipped != 0.0:
            tilts[ticker] = clipped
    validated["per_ticker_tilts"] = tilts

    # --- excluded_tickers ---
    raw_excluded = proposed.get("excluded_tickers", []) or []
    if not isinstance(raw_excluded, list):
        corrections.append("excluded_tickers was not a list; ignored entirely")
        raw_excluded = []
    excluded = []
    for ticker in raw_excluded:
        if ticker not in equity_tickers:
            corrections.append(f"excluded_tickers entry {ticker!r} is not a known equity; ignored")
            continue
        if ticker not in excluded:
            excluded.append(ticker)

    cap = max_excluded_tickers(len(equity_tickers))
    if len(excluded) > cap:
        corrections.append(f"excluded_tickers proposed {len(excluded)} tickers, "
                            f"more than the allowed maximum of {cap}; truncated to the first {cap}")
        excluded = excluded[:cap]
    validated["excluded_tickers"] = excluded

    # --- risk_level ---
    risk_level = proposed.get("risk_level", current.get("risk_level", DEFAULT_RISK_LEVEL))
    if risk_level not in RISK_LEVELS:
        corrections.append(f"risk_level {risk_level!r} not one of {RISK_LEVELS}; "
                            f"kept current value {current.get('risk_level', DEFAULT_RISK_LEVEL)!r}")
        risk_level = current.get("risk_level", DEFAULT_RISK_LEVEL)
    validated["risk_level"] = risk_level

    return validated, corrections
