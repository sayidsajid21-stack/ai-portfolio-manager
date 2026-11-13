from report_lib import (
    format_weights_table, format_weight_changes, format_news_section,
    format_performance_section, format_orders_section, build_report,
    format_strategy_settings_section,
)


class _FakeNewsScore:
    def __init__(self, sentiment, event_type, confidence, rationale):
        self.sentiment = sentiment
        self.event_type = event_type
        self.confidence = confidence
        self.rationale = rationale


def test_format_weights_table_includes_all_tickers_and_totals():
    weights = {"AAPL": 0.40, "AGG": 0.60}
    table = format_weights_table(weights, ["AAPL", "AGG"], ["AAPL"], ["AGG"])
    assert "AAPL" in table
    assert "40.00%" in table
    assert "Total equity" in table
    assert "Total bonds" in table
    assert "60.00%" in table


def test_format_weights_table_handles_missing_ticker_as_zero():
    weights = {"AAPL": 1.0}
    table = format_weights_table(weights, ["AAPL", "AGG"], ["AAPL"], ["AGG"])
    assert "0.00%" in table  # AGG not in weights dict


def test_format_weight_changes_handles_no_previous_recommendation():
    result = format_weight_changes({"AAPL": 0.5}, previous=None)
    assert "first report" in result.lower()


def test_format_weight_changes_reports_only_material_changes():
    current = {"AAPL": 0.30, "MSFT": 0.201}
    previous = {"AAPL": 0.10, "MSFT": 0.20}  # MSFT moved by 0.001 (below threshold)
    result = format_weight_changes(current, previous, min_change=0.005)
    assert "AAPL" in result
    assert "MSFT" not in result


def test_format_weight_changes_no_material_changes_message():
    current = {"AAPL": 0.300}
    previous = {"AAPL": 0.301}
    result = format_weight_changes(current, previous, min_change=0.005)
    assert "no material changes" in result.lower()


def test_format_weight_changes_shows_up_and_down_arrows():
    current = {"AAPL": 0.30, "MSFT": 0.10}
    previous = {"AAPL": 0.10, "MSFT": 0.30}
    result = format_weight_changes(current, previous, min_change=0.005)
    assert "▲" in result  # AAPL went up
    assert "▼" in result  # MSFT went down


def test_format_news_section_empty():
    result = format_news_section({})
    assert "no news adjustments" in result.lower()


def test_format_news_section_lists_scores():
    scores = {"AAPL": _FakeNewsScore(0.5, "earnings", 0.8, "Beat estimates.")}
    result = format_news_section(scores)
    assert "AAPL" in result
    assert "+0.50" in result
    assert "earnings" in result
    assert "Beat estimates." in result


def test_format_performance_section_beats_baseline():
    result = format_performance_section(0.08, 0.05, old_model_return=0.06)
    assert "beat the equal-weight baseline" in result
    assert "✅" in result


def test_format_performance_section_does_not_beat_baseline():
    result = format_performance_section(0.03, 0.05, old_model_return=None)
    assert "did NOT beat" in result
    assert "⚠️" in result


def test_format_orders_section_not_configured():
    result = format_orders_section([], dry_run=True, alpaca_configured=False)
    assert "not configured" in result.lower()


def test_format_orders_section_no_orders_needed():
    result = format_orders_section([], dry_run=True, alpaca_configured=True)
    assert "no rebalancing orders needed" in result.lower()


def test_format_orders_section_lists_dry_run_orders():
    orders = [{"side": "buy", "symbol": "AAPL", "notional": 1234.56, "note": "dry run — not submitted"}]
    result = format_orders_section(orders, dry_run=True, alpaca_configured=True)
    assert "DRY RUN" in result
    assert "AAPL" in result
    assert "1,234.56" in result


def test_build_report_includes_all_sections():
    report = build_report(
        report_date="2026-01-01",
        current_weights={"AAPL": 0.75, "AGG": 0.25},
        previous_weights={"AAPL": 0.70, "AGG": 0.30},
        news_scores={"AAPL": _FakeNewsScore(0.2, "product", 0.6, "New product launch.")},
        new_model_return=0.05,
        baseline_return=0.03,
        old_model_return=0.04,
        orders=[{"side": "buy", "symbol": "AAPL", "notional": 500.0, "status": "accepted"}],
        dry_run=False,
        alpaca_configured=True,
        all_tickers=["AAPL", "AGG"],
        equity_tickers=["AAPL"],
        bond_tickers=["AGG"],
    )
    assert "2026-01-01" in report
    assert "Recommended allocation" in report
    assert "Changes since last month" in report
    assert "News sentiment adjustments" in report
    assert "Performance vs. baseline" in report
    assert "Paper trading orders" in report
    assert "AAPL" in report


def test_format_strategy_settings_section_handles_none():
    result = format_strategy_settings_section(None)
    assert "using defaults" in result.lower()


def test_format_strategy_settings_section_shows_tilts_and_exclusions():
    settings = {
        "equity_share": 0.70,
        "per_ticker_tilts": {"AAPL": 0.03, "KO": -0.02},
        "excluded_tickers": ["XOM"],
        "risk_level": "conservative",
    }
    result = format_strategy_settings_section(settings)
    assert "70%" in result
    assert "30%" in result  # bond share = 1 - 0.70
    assert "conservative" in result
    assert "AAPL" in result and "+3.0%" in result
    assert "XOM" in result


def test_format_strategy_settings_section_shows_none_for_empty_tilts_and_exclusions():
    settings = {"equity_share": 0.75, "per_ticker_tilts": {}, "excluded_tickers": [], "risk_level": "balanced"}
    result = format_strategy_settings_section(settings)
    assert "none" in result.lower()


def test_build_report_includes_strategy_settings_section():
    report = build_report(
        report_date="2026-01-01", current_weights={"AAPL": 0.75, "AGG": 0.25}, previous_weights=None,
        news_scores={}, new_model_return=0.05, baseline_return=0.03, old_model_return=None,
        orders=[], dry_run=True, alpaca_configured=False,
        all_tickers=["AAPL", "AGG"], equity_tickers=["AAPL"], bond_tickers=["AGG"],
        strategy_settings={"equity_share": 0.75, "per_ticker_tilts": {}, "excluded_tickers": [], "risk_level": "balanced"},
    )
    assert "Strategy settings in effect" in report
    assert "balanced" in report
