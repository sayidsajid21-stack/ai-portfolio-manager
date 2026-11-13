"""
Strategy review agent (Phase 5).

Reads recent performance, news, and the current strategy settings, asks
Claude for a proposed settings change via structured output, validates
and clips the proposal against the hard limits in strategy_config.py,
and — if anything actually changed — writes the new settings.json and
logs the full rationale. Opening a pull request for human approval is
the workflow's job (.github/workflows/strategy_review.yml), not this
script's: this script only ever writes a new strategy_settings.json to
the working tree; it never commits, pushes, or calls the GitHub API
itself, and never applies anything automatically.

This is a no-op (prints a message, exits) unless a Claude credential is
configured, same opt-in pattern as news_sentiment.py — see README for
cost and setup.
"""
import json
import os
from datetime import datetime, timezone

from typing import Dict, List

from pydantic import BaseModel, Field

import strategy_config

DEFAULT_MODEL = "claude-opus-5-5"
REVIEW_LOG_PATH = "strategy_review_log.jsonl"
NEWS_LOG_PATH = "news_log.jsonl"
METADATA_PATH = "model_metadata.json"
RECENT_NEWS_LINES = 50


class StrategyProposal(BaseModel):
    """Structured output schema for one strategy review."""
    equity_share: float = Field(
        description="Proposed fraction of the portfolio in equities (vs. bonds), "
                    "between 0.50 and 0.85. Keep the current value if no change is warranted."
    )
    per_ticker_tilts: Dict[str, float] = Field(
        default_factory=dict,
        description="Ticker -> a persistent strategic tilt between -0.05 and 0.05. "
                    "Omit a ticker entirely if it should have no tilt. This is NOT "
                    "for daily news reactions — only for a deliberate, lasting lean."
    )
    excluded_tickers: List[str] = Field(
        default_factory=list,
        description="Tickers to exclude entirely from the portfolio (e.g. a serious "
                    "legal, regulatory, or going-concern risk) until this is revisited.",
    )
    risk_level: str = Field(description="One of: conservative, balanced, aggressive.")
    rationale: str = Field(description="2-4 sentences explaining the reasoning for this proposal.")


def _read_recent_news_summary(path: str = NEWS_LOG_PATH, max_lines: int = RECENT_NEWS_LINES) -> str:
    if not os.path.exists(path):
        return "(no news log available)"
    with open(path) as f:
        lines = f.readlines()[-max_lines:]
    if not lines:
        return "(no news log available)"
    return "".join(lines)


def _read_recent_performance(path: str = METADATA_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_review_prompt(current_settings: dict, performance: dict, news_summary: str, equity_tickers) -> str:
    return (
        "You are reviewing the settings of a long-term-investing (not day-trading) "
        "portfolio strategy. Propose whether any of the following should change, "
        "and explain why. Be conservative — only propose a change when there's a "
        "real reason to, not every time.\n\n"
        f"Equity universe: {', '.join(equity_tickers)}\n\n"
        f"Current settings: {json.dumps(current_settings, indent=2)}\n\n"
        f"Recent model performance (model_metadata.json): {json.dumps(performance, indent=2)}\n\n"
        f"Recent news log entries (most recent {RECENT_NEWS_LINES} lines, JSONL):\n{news_summary}\n"
    )


def propose_strategy_update(current_settings: dict, performance: dict, news_summary: str,
                             equity_tickers, client=None, model: str = DEFAULT_MODEL) -> StrategyProposal:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    prompt = build_review_prompt(current_settings, performance, news_summary, equity_tickers)
    response = client.messages.parse(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        output_format=StrategyProposal,
    )
    return response.parsed_output


def log_review(current_settings: dict, validated_settings: dict, corrections: list, rationale: str,
                changed: bool, path: str = REVIEW_LOG_PATH):
    entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "current_settings": current_settings,
        "proposed_settings": validated_settings,
        "corrections": corrections,
        "rationale": rationale,
        "changed": changed,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_review(equity_tickers, client=None) -> tuple:
    """
    Runs one full review cycle. Returns (changed: bool, validated_settings: dict).
    If `changed` is True, strategy_config.SETTINGS_PATH has already been
    overwritten with `validated_settings` — the workflow is responsible
    for turning that file change into a pull request, never applying it
    directly to a running system.
    """
    current_settings = strategy_config.load_settings()
    performance = _read_recent_performance()
    news_summary = _read_recent_news_summary()

    try:
        proposal = propose_strategy_update(current_settings, performance, news_summary, equity_tickers, client=client)
    except Exception as e:
        print(f"Strategy review skipped (no usable Claude credential, or the call failed): {e}")
        return False, current_settings

    validated_settings, corrections = strategy_config.validate_and_clip(
        proposal.model_dump(), current_settings, equity_tickers
    )

    changed = validated_settings != current_settings
    log_review(current_settings, validated_settings, corrections, proposal.rationale, changed)

    if corrections:
        print("Corrections applied to the proposal:")
        for c in corrections:
            print(f"  - {c}")

    if changed:
        strategy_config.save_settings(validated_settings)
        print(f"Strategy settings changed. Rationale: {proposal.rationale}")
    else:
        print("No strategy settings change proposed (or the validated proposal matches current settings).")

    return changed, validated_settings


if __name__ == "__main__":
    import portfolio_lib
    run_review(portfolio_lib.EQUITY_TICKERS)
