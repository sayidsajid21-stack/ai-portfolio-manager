"""
News sentiment overlay for the monthly recommendation (Phase 4).

Headlines come from yfinance's free `Ticker.news` — no API key needed,
but it only returns a handful of *recent* headlines with essentially no
history, so this cannot be backtested the way price/fundamentals data
can be (see README's "News sentiment overlay" section). For that reason
this is a *live* adjustment applied only to today's final recommendation
in rebalance.py — it is NOT fed into the RL model as a training feature,
since there's no real historical news dataset to train on yet.

Scoring each ticker's headlines is one Claude API call per ticker
(structured JSON output, so results are typed and validated, not just
"hopefully formatted text"). This costs a small amount of real money per
run — see README for an estimate and how to enable/disable it.

If ANTHROPIC_API_KEY (or another credential anthropic.Anthropic() can
resolve) is not configured, build_news_scores() returns an empty dict
for every ticker and the overlay is a no-op — rebalance.py's weights are
unaffected. This is deliberate: news scoring is opt-in, gated by whether
you've chosen to set the credential, not something this module decides
on its own.
"""
import json
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-5-5"
MAX_HEADLINES_PER_TICKER = 5
NEWS_LOG_PATH = "news_log.jsonl"

# Maximum weight adjustment a single ticker's news score can apply, as a
# fraction of that ticker's existing weight (see
# portfolio_lib.apply_news_adjustment). Kept small and symmetric: this is
# a nudge based on a handful of recent headlines, not a reason to make a
# large allocation change.
MAX_ADJUSTMENT_FRACTION = 0.03


class NewsScore(BaseModel):
    """Structured output schema for one ticker's news sentiment."""
    sentiment: float = Field(
        description="Overall sentiment from -1.0 (very negative) to 1.0 (very positive), "
                    "0.0 if neutral or headlines are routine/non-material."
    )
    event_type: str = Field(
        description="One of: earnings, guidance, legal, macro, management, product, "
                    "mergers_acquisitions, routine, other."
    )
    confidence: float = Field(description="0.0 to 1.0: how confident this assessment is.")
    horizon: str = Field(
        description="One of: short_term (news likely to matter for days/weeks), "
                    "long_term (likely to matter for months+), unclear."
    )
    rationale: str = Field(description="One sentence explaining the sentiment score.")


def fetch_headlines(ticker: str, yf_ticker=None, max_headlines: int = MAX_HEADLINES_PER_TICKER) -> list:
    """
    Fetch recent headlines for a ticker via yfinance (free, no API key).
    `yf_ticker` is injectable for testing; defaults to a real
    yfinance.Ticker(ticker).

    yfinance's `.news` shape has varied across versions — some return a
    flat list of dicts with 'title'/'publisher' keys directly, others
    nest those under a 'content' key. This normalizes both.
    """
    if yf_ticker is None:
        import yfinance as yf
        yf_ticker = yf.Ticker(ticker)

    raw_news = yf_ticker.news or []
    headlines = []
    for item in raw_news[:max_headlines]:
        content = item.get("content", item)  # unwrap newer yfinance shape if present
        title = content.get("title")
        if not title:
            continue
        headlines.append({
            "title": title,
            "publisher": (content.get("provider") or {}).get("displayName")
                         if isinstance(content.get("provider"), dict)
                         else content.get("publisher"),
        })
    return headlines


def score_ticker_news(ticker: str, headlines: list, client=None, model: str = DEFAULT_MODEL) -> NewsScore:
    """
    Score one ticker's headlines with Claude, using structured output so
    the result is a validated NewsScore rather than free-form text. If
    `client` is None, resolves credentials the normal anthropic.Anthropic()
    way (env var or `ant auth login` profile) — pass a fake client in
    tests to avoid any real network/API call.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    headlines_text = "\n".join(f"- {h['title']} ({h.get('publisher', 'unknown source')})" for h in headlines)
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Here are recent headlines about {ticker}:\n\n{headlines_text}\n\n"
                "Assess the sentiment and likely materiality of this news for a "
                "long-term investor (not a day trader) holding this stock."
            ),
        }],
        output_format=NewsScore,
    )
    return response.parsed_output


def build_news_scores(tickers, client=None, yf_ticker_factory=None) -> dict:
    """
    Fetch headlines and score them for each ticker in `tickers`.

    Returns {ticker: NewsScore}. A ticker is omitted from the result
    (not given a zero/neutral placeholder) if it has no headlines or
    scoring fails for it — rebalance.py treats a missing entry as "no
    adjustment", not as a signal.

    If no Claude credential is available at all, catches the resulting
    AuthenticationError on the first attempt and returns {} immediately
    rather than failing once per ticker.
    """
    scores = {}
    client_checked = client is not None

    for i, ticker in enumerate(tickers):
        try:
            yf_ticker = yf_ticker_factory(ticker) if yf_ticker_factory else None
            headlines = fetch_headlines(ticker, yf_ticker=yf_ticker)
            if not headlines:
                continue
            scores[ticker] = score_ticker_news(ticker, headlines, client=client)
            client_checked = True
        except Exception as e:
            print(f"  Warning: news scoring failed for {ticker}: {e}")
            if not client_checked:
                # Credential problem, not a per-ticker problem — no point
                # retrying for every remaining ticker.
                print("  Skipping news sentiment overlay for this run (no usable Claude credential).")
                break

    return scores


def log_news_scores(scores: dict, path: str = NEWS_LOG_PATH):
    """
    Append today's news scores to a JSONL log, one line per ticker, so
    there's a growing real history to eventually backtest against (even
    though yfinance's own headline history is too short to backtest
    today). Each line is a complete, independent JSON object.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(path, "a") as f:
        for ticker, score in scores.items():
            f.write(json.dumps({"date": today, "ticker": ticker, **score.model_dump()}) + "\n")
