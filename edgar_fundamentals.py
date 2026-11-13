"""
Point-in-time fundamentals from SEC EDGAR's free XBRL company-facts API.

This replaces the Phase 0 simplification in portfolio_lib.build_features,
where a single current fundamentals snapshot was broadcast across the
entire training history. Here, each historical date gets the P/E,
price-to-book, and dividend-yield figures that were actually knowable on
that date — computed from the filing's `filed` date (when the fact became
public), not its `end` date (the fiscal period it describes), so there is
no lookahead: a Q2 EPS figure doesn't "count" until the 10-Q that reports
it was actually filed.

No API key is needed, but SEC's Fair Access policy requires a descriptive
User-Agent identifying the requester (ideally with a contact email) on
every request. Set it via the EDGAR_USER_AGENT environment variable; see
README for how to configure this as a GitHub Actions secret or repo
variable. Requests are also rate-limited by SEC to ~10 req/sec, so `_get`
adds a small delay between calls.

Known simplifications (documented, not yet fixed):
  - EPS uses the annual (10-K) figure only, updated once a year, rather
    than a rolling trailing-twelve-month figure built from quarterly
    10-Qs. This is simpler and still removes the random-data and
    lookahead bugs; a TTM refinement is a natural follow-up.
  - Dividend yield annualizes the most recently declared quarterly
    dividend (x4) rather than summing the actual trailing four
    quarters, which slightly misstates yield right after a dividend
    change.
"""
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

DEFAULT_USER_AGENT = (
    "ai-portfolio-manager research tool "
    "(set EDGAR_USER_AGENT env var to your own contact email per SEC's Fair Access policy: "
    "https://www.sec.gov/os/webmaster-faq#developers)"
)

# Neutral fallback values used when no SEC data is available yet for a
# given date (e.g. before a company's first filing in our window) — the
# same defaults Phase 0 used for "unknown" fundamentals, not a guess about
# the future.
DEFAULT_PE = 20.0
DEFAULT_P_TO_IV = 1.0
DEFAULT_DIV = 0.0

_last_request_time = [0.0]


def _get(url, session=None):
    """GET with the required SEC User-Agent header and a light rate limit."""
    import requests
    session = session or requests
    headers = {"User-Agent": os.environ.get("EDGAR_USER_AGENT", DEFAULT_USER_AGENT)}

    # SEC asks for no more than ~10 requests/second.
    elapsed = time.time() - _last_request_time[0]
    if elapsed < 0.11:
        time.sleep(0.11 - elapsed)
    response = session.get(url, headers=headers, timeout=30)
    _last_request_time[0] = time.time()
    response.raise_for_status()
    return response.json()


def get_cik_for_ticker(ticker: str, session=None, _cache={}) -> int:
    """
    Resolve a ticker to its SEC CIK number using SEC's own ticker map.
    Cached in-process since the map covers every ticker in one request.
    """
    if not _cache:
        data = _get(SEC_TICKER_MAP_URL, session=session)
        for entry in data.values():
            _cache[entry["ticker"].upper()] = entry["cik_str"]
    if ticker.upper() not in _cache:
        raise KeyError(f"No SEC CIK found for ticker {ticker!r}")
    return _cache[ticker.upper()]


def fetch_company_facts(cik: int, session=None) -> dict:
    """Fetch the full XBRL company-facts JSON for a given CIK."""
    return _get(SEC_FACTS_URL.format(cik=cik), session=session)


def parse_point_in_time_fact(facts_json: dict, tag: str, taxonomy: str = "us-gaap", forms=None):
    """
    Extract the raw (filed, end, val, form) entries for one XBRL tag from
    a company-facts JSON blob. Returns a list of dicts, unsorted.
    `forms`, if given, restricts to specific form types (e.g. ('10-K',)).
    """
    try:
        units = facts_json["facts"][taxonomy][tag]["units"]
    except KeyError:
        return []

    entries = []
    for unit_values in units.values():
        for entry in unit_values:
            if forms is not None and entry.get("form") not in forms:
                continue
            if "filed" not in entry or "val" not in entry:
                continue
            entries.append({
                "filed": entry["filed"],
                "end": entry.get("end"),
                "val": entry["val"],
                "form": entry.get("form"),
            })
    return entries


def to_point_in_time_series(entries: list) -> pd.Series:
    """
    Convert parsed fact entries into a Series indexed by filing date (the
    date the value became public), sorted ascending, de-duplicated so
    each filed date keeps its last-reported value.
    """
    if not entries:
        return pd.Series(dtype=float)
    df = pd.DataFrame(entries)
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values("filed").drop_duplicates("filed", keep="last")
    return df.set_index("filed")["val"]


def align_to_price_dates(series: pd.Series, price_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a point-in-time series onto a target date index: each
    price date gets the most recently *filed* value as of that date, and
    NaN before the series' first filing (no data was public yet).
    """
    if series.empty:
        return pd.Series(np.nan, index=price_dates)
    combined_index = series.index.union(price_dates).sort_values()
    return series.reindex(combined_index).ffill().reindex(price_dates)


def build_point_in_time_fundamentals(ticker: str, price_dates: pd.DatetimeIndex, session=None) -> pd.DataFrame:
    """
    Build a DataFrame of point-in-time `pe`, `p_to_iv`, and `div` columns
    for one ticker, aligned to `price_dates`. Intended to be passed into
    portfolio_lib.build_features in place of a constant snapshot dict —
    see that function's docstring for how the two modes differ.

    Any date for which SEC has not yet published the relevant filing gets
    the neutral defaults (DEFAULT_PE / DEFAULT_P_TO_IV / DEFAULT_DIV)
    rather than NaN, so the resulting DataFrame is immediately usable
    without an extra dropna pass specific to fundamentals (price-return
    data can still require its own dropna for other reasons, e.g.
    momentum windows).
    """
    cik = get_cik_for_ticker(ticker, session=session)
    facts = fetch_company_facts(cik, session=session)

    eps_series = to_point_in_time_series(
        parse_point_in_time_fact(facts, "EarningsPerShareBasic", forms=("10-K",))
    )
    equity_series = to_point_in_time_series(
        parse_point_in_time_fact(facts, "StockholdersEquity")
    )
    shares_series = to_point_in_time_series(
        parse_point_in_time_fact(facts, "CommonStockSharesOutstanding")
    )
    div_series = to_point_in_time_series(
        parse_point_in_time_fact(facts, "CommonStockDividendsPerShareDeclared")
    )

    eps_aligned = align_to_price_dates(eps_series, price_dates)
    equity_aligned = align_to_price_dates(equity_series, price_dates)
    shares_aligned = align_to_price_dates(shares_series, price_dates)
    div_aligned = align_to_price_dates(div_series, price_dates)

    book_value_per_share = equity_aligned / shares_aligned.replace(0, np.nan)

    return pd.DataFrame({
        "eps": eps_aligned,
        "book_value_per_share": book_value_per_share,
        "declared_div_per_share": div_aligned,
    }, index=price_dates)


def fundamentals_to_pe_iv_div(fundamentals_df: pd.DataFrame, price_close: pd.Series) -> pd.DataFrame:
    """
    Turn the raw point-in-time fundamentals (eps, book value per share,
    declared dividend per share) into the pe/p_to_iv/div columns
    portfolio_lib.build_features expects, using the same price series the
    fundamentals will be paired with.
    """
    eps = fundamentals_df["eps"]
    book_value_per_share = fundamentals_df["book_value_per_share"]
    declared_div = fundamentals_df["declared_div_per_share"]

    pe = price_close / eps.where(eps > 0)
    p_to_iv = price_close / book_value_per_share.where(book_value_per_share > 0)
    div_yield = (declared_div * 4) / price_close

    pe = pe.clip(1.0, 100.0).fillna(DEFAULT_PE)
    p_to_iv = p_to_iv.clip(0.1, 10.0).fillna(DEFAULT_P_TO_IV)
    div_yield = div_yield.clip(0.0, 0.15).fillna(DEFAULT_DIV)

    return pd.DataFrame({"pe": pe, "p_to_iv": p_to_iv, "div": div_yield})
