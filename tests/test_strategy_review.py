import json
import os

import pytest

import strategy_config
import strategy_review
from strategy_review import StrategyProposal, run_review, build_review_prompt

EQUITY_TICKERS = ["AAA", "BBB", "CCC", "DDD"]


class _FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, result):
        self._result = result

    def parse(self, **kwargs):
        return _FakeParsedResponse(self._result)


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)


class _FailingMessages:
    def parse(self, **kwargs):
        raise RuntimeError("AuthenticationError: no credential")


class _FailingClient:
    messages = _FailingMessages()


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_build_review_prompt_includes_key_context():
    prompt = build_review_prompt(
        current_settings={"equity_share": 0.75}, performance={"holdout_return": 0.05},
        news_summary="(no news log available)", equity_tickers=EQUITY_TICKERS,
    )
    assert "AAA" in prompt
    assert "0.75" in prompt
    assert "holdout_return" in prompt


def test_run_review_no_op_when_credential_fails():
    changed, settings = run_review(EQUITY_TICKERS, client=_FailingClient())
    assert changed is False
    assert settings == strategy_config.DEFAULT_SETTINGS
    # no settings file should have been written
    assert not os.path.exists(strategy_config.SETTINGS_PATH)


def test_run_review_writes_settings_file_when_changed():
    proposal = StrategyProposal(
        equity_share=0.70, per_ticker_tilts={"AAA": 0.03}, excluded_tickers=[],
        risk_level="conservative", rationale="Reducing equity exposure given recent volatility.",
    )
    client = _FakeClient(proposal)
    changed, settings = run_review(EQUITY_TICKERS, client=client)

    assert changed is True
    assert settings["equity_share"] == 0.70
    assert settings["per_ticker_tilts"] == {"AAA": 0.03}
    assert settings["risk_level"] == "conservative"

    # the settings file should actually have been written to disk
    assert os.path.exists(strategy_config.SETTINGS_PATH)
    with open(strategy_config.SETTINGS_PATH) as f:
        on_disk = json.load(f)
    assert on_disk == settings


def test_run_review_does_not_write_file_when_proposal_matches_current():
    current = strategy_config.DEFAULT_SETTINGS
    proposal = StrategyProposal(
        equity_share=current["equity_share"], per_ticker_tilts={}, excluded_tickers=[],
        risk_level=current["risk_level"], rationale="Everything looks fine, no change needed.",
    )
    client = _FakeClient(proposal)
    changed, settings = run_review(EQUITY_TICKERS, client=client)

    assert changed is False
    assert not os.path.exists(strategy_config.SETTINGS_PATH)


def test_run_review_clips_an_out_of_range_proposal_before_writing():
    proposal = StrategyProposal(
        equity_share=0.99,  # outside the allowed [0.50, 0.85] range
        per_ticker_tilts={}, excluded_tickers=[], risk_level="balanced",
        rationale="Going all-in on equities.",
    )
    client = _FakeClient(proposal)
    changed, settings = run_review(EQUITY_TICKERS, client=client)

    assert changed is True
    assert settings["equity_share"] == strategy_config.EQUITY_SHARE_RANGE[1]  # clipped, not 0.99


def test_run_review_logs_every_run_including_no_op(tmp_path):
    proposal = StrategyProposal(
        equity_share=strategy_config.DEFAULT_SETTINGS["equity_share"],
        per_ticker_tilts={}, excluded_tickers=[],
        risk_level=strategy_config.DEFAULT_SETTINGS["risk_level"],
        rationale="No change needed this month.",
    )
    client = _FakeClient(proposal)
    run_review(EQUITY_TICKERS, client=client)

    assert os.path.exists(strategy_review.REVIEW_LOG_PATH)
    with open(strategy_review.REVIEW_LOG_PATH) as f:
        entries = [json.loads(line) for line in f]
    assert len(entries) == 1
    assert entries[0]["changed"] is False
    assert entries[0]["rationale"] == "No change needed this month."


def test_run_review_logs_rationale_and_corrections_when_clipped():
    proposal = StrategyProposal(
        equity_share=0.99, per_ticker_tilts={"ZZZ": 0.5}, excluded_tickers=[],
        risk_level="balanced", rationale="Aggressive stance.",
    )
    client = _FakeClient(proposal)
    run_review(EQUITY_TICKERS, client=client)

    with open(strategy_review.REVIEW_LOG_PATH) as f:
        entry = json.loads(f.readline())
    assert entry["rationale"] == "Aggressive stance."
    assert len(entry["corrections"]) >= 1  # equity_share clip + unknown ticker drop
