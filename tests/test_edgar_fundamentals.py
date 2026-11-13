import pandas as pd
import numpy as np
import pytest

from edgar_fundamentals import (
    parse_point_in_time_fact, to_point_in_time_series, align_to_price_dates,
    fundamentals_to_pe_iv_div, get_cik_for_ticker, build_point_in_time_fundamentals,
    DEFAULT_PE, DEFAULT_P_TO_IV, DEFAULT_DIV,
)


def _facts_json():
    """A synthetic company-facts JSON matching SEC EDGAR's real shape
    closely enough to exercise the parsing logic, without any network
    access."""
    return {
        "facts": {
            "us-gaap": {
                "EarningsPerShareBasic": {
                    "units": {
                        "USD/shares": [
                            {"end": "2022-12-31", "val": 4.00, "form": "10-K", "filed": "2023-02-15"},
                            {"end": "2023-03-31", "val": 1.10, "form": "10-Q", "filed": "2023-05-01"},
                            {"end": "2023-12-31", "val": 5.00, "form": "10-K", "filed": "2024-02-14"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "val": 1_000_000, "form": "10-Q", "filed": "2023-05-01"},
                            {"end": "2023-12-31", "val": 1_200_000, "form": "10-K", "filed": "2024-02-14"},
                        ]
                    }
                },
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2023-03-31", "val": 100_000, "form": "10-Q", "filed": "2023-05-01"},
                            {"end": "2023-12-31", "val": 100_000, "form": "10-K", "filed": "2024-02-14"},
                        ]
                    }
                },
                "CommonStockDividendsPerShareDeclared": {
                    "units": {
                        "USD/shares": [
                            {"end": "2023-03-31", "val": 0.20, "form": "10-Q", "filed": "2023-05-01"},
                            {"end": "2023-12-31", "val": 0.25, "form": "10-K", "filed": "2024-02-14"},
                        ]
                    }
                },
            }
        }
    }


def test_parse_point_in_time_fact_filters_by_form():
    facts = _facts_json()
    all_entries = parse_point_in_time_fact(facts, "EarningsPerShareBasic")
    assert len(all_entries) == 3
    only_10k = parse_point_in_time_fact(facts, "EarningsPerShareBasic", forms=("10-K",))
    assert len(only_10k) == 2
    assert all(e["form"] == "10-K" for e in only_10k)


def test_parse_point_in_time_fact_missing_tag_returns_empty():
    facts = _facts_json()
    assert parse_point_in_time_fact(facts, "SomeTagThatDoesNotExist") == []


def test_to_point_in_time_series_indexed_by_filed_date_not_end_date():
    facts = _facts_json()
    entries = parse_point_in_time_fact(facts, "EarningsPerShareBasic", forms=("10-K",))
    series = to_point_in_time_series(entries)
    # indexed by *filed* date, not *end* date — this is the lookahead fix
    assert pd.Timestamp("2023-02-15") in series.index
    assert series.loc["2023-02-15"] == 4.00
    assert pd.Timestamp("2024-02-14") in series.index
    assert series.loc["2024-02-14"] == 5.00
    # the fiscal period end dates should NOT appear as index values
    assert pd.Timestamp("2022-12-31") not in series.index


def test_align_to_price_dates_forward_fills_and_has_no_lookahead():
    facts = _facts_json()
    entries = parse_point_in_time_fact(facts, "EarningsPerShareBasic", forms=("10-K",))
    series = to_point_in_time_series(entries)

    price_dates = pd.date_range("2023-01-01", "2024-03-01", freq="MS")
    aligned = align_to_price_dates(series, price_dates)

    # Before the first 10-K was filed (2023-02-15), there's no data yet.
    assert pd.isna(aligned.loc["2023-01-01"])
    # On/after 2023-02-15 but before the next filing, value is the first EPS.
    assert aligned.loc["2023-03-01"] == 4.00
    # After the second 10-K was filed (2024-02-14), value updates.
    assert aligned.loc["2024-03-01"] == 5.00


def test_fundamentals_to_pe_iv_div_computes_correct_ratios():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    fundamentals_df = pd.DataFrame({
        "eps": [5.0, 5.0, 5.0],
        "book_value_per_share": [10.0, 10.0, 10.0],
        "declared_div_per_share": [0.25, 0.25, 0.25],
    }, index=dates)
    price = pd.Series([100.0, 110.0, 50.0], index=dates)

    result = fundamentals_to_pe_iv_div(fundamentals_df, price)

    # pe = price / eps
    assert result["pe"].iloc[0] == pytest.approx(20.0)  # 100/5
    assert result["pe"].iloc[1] == pytest.approx(22.0)  # 110/5
    # p_to_iv = price / book value per share
    assert result["p_to_iv"].iloc[0] == pytest.approx(10.0)  # 100/10
    # div yield = (declared * 4) / price
    assert result["div"].iloc[0] == pytest.approx(1.0 / 100.0)  # (0.25*4)/100


def test_fundamentals_to_pe_iv_div_handles_missing_data_with_neutral_defaults():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    fundamentals_df = pd.DataFrame({
        "eps": [np.nan, np.nan],
        "book_value_per_share": [np.nan, np.nan],
        "declared_div_per_share": [np.nan, np.nan],
    }, index=dates)
    price = pd.Series([100.0, 100.0], index=dates)

    result = fundamentals_to_pe_iv_div(fundamentals_df, price)
    assert (result["pe"] == DEFAULT_PE).all()
    assert (result["p_to_iv"] == DEFAULT_P_TO_IV).all()
    assert (result["div"] == DEFAULT_DIV).all()


def test_fundamentals_to_pe_iv_div_ignores_negative_or_zero_eps():
    # A loss-making quarter (negative EPS) shouldn't produce a negative
    # or nonsensical P/E — it should fall back to the neutral default.
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    fundamentals_df = pd.DataFrame({
        "eps": [-2.0],
        "book_value_per_share": [10.0],
        "declared_div_per_share": [0.0],
    }, index=dates)
    price = pd.Series([100.0], index=dates)

    result = fundamentals_to_pe_iv_div(fundamentals_df, price)
    assert result["pe"].iloc[0] == DEFAULT_PE


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Routes .get() calls to canned JSON based on the URL, so
    build_point_in_time_fundamentals can be exercised with no real
    network access."""

    def __init__(self, ticker_map, facts_by_cik):
        self.ticker_map = ticker_map
        self.facts_by_cik = facts_by_cik

    def get(self, url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _FakeResponse(self.ticker_map)
        for cik, facts in self.facts_by_cik.items():
            if f"CIK{cik:010d}" in url:
                return _FakeResponse(facts)
        raise AssertionError(f"Unexpected URL requested: {url}")


def test_get_cik_for_ticker_uses_injected_cache_and_session():
    ticker_map = {"0": {"ticker": "ZZZ", "cik_str": 42}}
    session = _FakeSession(ticker_map, facts_by_cik={})
    cik = get_cik_for_ticker("ZZZ", session=session, _cache={})
    assert cik == 42


def test_get_cik_for_ticker_raises_for_unknown_ticker():
    ticker_map = {"0": {"ticker": "ZZZ", "cik_str": 42}}
    session = _FakeSession(ticker_map, facts_by_cik={})
    with pytest.raises(KeyError):
        get_cik_for_ticker("NOTATICKER", session=session, _cache={})


def test_build_point_in_time_fundamentals_end_to_end_with_fake_session():
    ticker_map = {"0": {"ticker": "ZZZ", "cik_str": 42}}
    session = _FakeSession(ticker_map, facts_by_cik={42: _facts_json()})

    price_dates = pd.date_range("2023-01-01", "2024-03-01", freq="MS")
    result = build_point_in_time_fundamentals("ZZZ", price_dates, session=session)

    assert list(result.columns) == ["eps", "book_value_per_share", "declared_div_per_share"]
    assert len(result) == len(price_dates)
    # Before any filing, eps should be NaN (no data yet, not a fabricated 0).
    assert pd.isna(result["eps"].iloc[0])
    # After the first 10-K filed 2023-02-15, eps should reflect it.
    march_2023 = pd.Timestamp("2023-03-01")
    assert result.loc[march_2023, "eps"] == 4.00
