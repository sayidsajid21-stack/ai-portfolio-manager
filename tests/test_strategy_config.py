import json

import pytest

from strategy_config import (
    load_settings, save_settings, validate_and_clip, max_excluded_tickers,
    DEFAULT_SETTINGS, EQUITY_SHARE_RANGE, MAX_TILT, RISK_LEVELS,
)

EQUITY_TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def test_load_settings_returns_defaults_when_file_missing(tmp_path):
    settings = load_settings(path=str(tmp_path / "does_not_exist.json"))
    assert settings == DEFAULT_SETTINGS


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "settings.json")
    settings = {"equity_share": 0.70, "per_ticker_tilts": {"AAA": 0.03},
                "excluded_tickers": ["BBB"], "risk_level": "conservative"}
    save_settings(settings, path=path)
    loaded = load_settings(path=path)
    assert loaded == settings


def test_load_settings_merges_onto_defaults_for_missing_keys(tmp_path):
    path = str(tmp_path / "partial.json")
    with open(path, "w") as f:
        json.dump({"equity_share": 0.60}, f)
    loaded = load_settings(path=path)
    assert loaded["equity_share"] == 0.60
    assert loaded["risk_level"] == DEFAULT_SETTINGS["risk_level"]  # filled in from defaults


def test_max_excluded_tickers_is_half_the_universe():
    assert max_excluded_tickers(10) == 5
    assert max_excluded_tickers(4) == 2


def test_validate_and_clip_passes_through_valid_proposal():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"equity_share": 0.70, "per_ticker_tilts": {"AAA": 0.02},
                "excluded_tickers": ["BBB"], "risk_level": "aggressive"}
    validated, corrections = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["equity_share"] == 0.70
    assert validated["per_ticker_tilts"] == {"AAA": 0.02}
    assert validated["excluded_tickers"] == ["BBB"]
    assert validated["risk_level"] == "aggressive"
    assert corrections == []


def test_validate_and_clip_clips_equity_share_above_range():
    current = dict(DEFAULT_SETTINGS)
    validated, corrections = validate_and_clip({"equity_share": 0.99}, current, EQUITY_TICKERS)
    assert validated["equity_share"] == EQUITY_SHARE_RANGE[1]
    assert len(corrections) == 1


def test_validate_and_clip_clips_equity_share_below_range():
    current = dict(DEFAULT_SETTINGS)
    validated, corrections = validate_and_clip({"equity_share": 0.10}, current, EQUITY_TICKERS)
    assert validated["equity_share"] == EQUITY_SHARE_RANGE[0]
    assert len(corrections) == 1


def test_validate_and_clip_handles_non_numeric_equity_share():
    current = dict(DEFAULT_SETTINGS)
    validated, corrections = validate_and_clip({"equity_share": "not a number"}, current, EQUITY_TICKERS)
    assert validated["equity_share"] == current["equity_share"]
    assert len(corrections) == 1


def test_validate_and_clip_drops_tilts_for_unknown_tickers():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"per_ticker_tilts": {"AAA": 0.02, "ZZZ": 0.03}}
    validated, corrections = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["per_ticker_tilts"] == {"AAA": 0.02}
    assert any("ZZZ" in c for c in corrections)


def test_validate_and_clip_clips_tilts_to_max_tilt():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"per_ticker_tilts": {"AAA": 1.0, "BBB": -1.0}}
    validated, corrections = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["per_ticker_tilts"]["AAA"] == MAX_TILT
    assert validated["per_ticker_tilts"]["BBB"] == -MAX_TILT
    assert len(corrections) == 2


def test_validate_and_clip_drops_zero_tilts_for_cleanliness():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"per_ticker_tilts": {"AAA": 0.0}}
    validated, _ = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["per_ticker_tilts"] == {}


def test_validate_and_clip_drops_excluded_unknown_tickers():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"excluded_tickers": ["AAA", "ZZZ"]}
    validated, corrections = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["excluded_tickers"] == ["AAA"]
    assert any("ZZZ" in c for c in corrections)


def test_validate_and_clip_truncates_excessive_exclusions():
    current = dict(DEFAULT_SETTINGS)
    # 4 equity tickers -> max_excluded_tickers = 2
    proposed = {"excluded_tickers": ["AAA", "BBB", "CCC"]}
    validated, corrections = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert len(validated["excluded_tickers"]) == 2
    assert len(corrections) == 1


def test_validate_and_clip_deduplicates_excluded_tickers():
    current = dict(DEFAULT_SETTINGS)
    proposed = {"excluded_tickers": ["AAA", "AAA"]}
    validated, _ = validate_and_clip(proposed, current, EQUITY_TICKERS)
    assert validated["excluded_tickers"] == ["AAA"]


def test_validate_and_clip_rejects_invalid_risk_level():
    current = dict(DEFAULT_SETTINGS)
    validated, corrections = validate_and_clip({"risk_level": "yolo"}, current, EQUITY_TICKERS)
    assert validated["risk_level"] == current["risk_level"]
    assert len(corrections) == 1


def test_validate_and_clip_accepts_each_valid_risk_level():
    current = dict(DEFAULT_SETTINGS)
    for level in RISK_LEVELS:
        validated, corrections = validate_and_clip({"risk_level": level}, current, EQUITY_TICKERS)
        assert validated["risk_level"] == level
        assert corrections == []


def test_validate_and_clip_missing_fields_fall_back_to_current():
    current = {"equity_share": 0.65, "per_ticker_tilts": {"AAA": 0.01},
               "excluded_tickers": ["BBB"], "risk_level": "conservative"}
    validated, corrections = validate_and_clip({}, current, EQUITY_TICKERS)
    assert validated["equity_share"] == 0.65
    assert validated["risk_level"] == "conservative"
    # empty-proposal fields (tilts/exclusions) are treated as "propose
    # none", not "keep current" — an empty proposal clears them
    assert validated["per_ticker_tilts"] == {}
    assert validated["excluded_tickers"] == []
