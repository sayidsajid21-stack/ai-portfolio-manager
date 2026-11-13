import json
import os

import pytest

from news_sentiment import (
    fetch_headlines, score_ticker_news, build_news_scores, log_news_scores, NewsScore,
)


class _FakeYfTicker:
    def __init__(self, news):
        self.news = news


def test_fetch_headlines_handles_flat_shape():
    yf_ticker = _FakeYfTicker(news=[
        {"title": "Company beats earnings", "publisher": "Reuters"},
        {"title": "New product launch", "publisher": "Bloomberg"},
    ])
    headlines = fetch_headlines("ZZZ", yf_ticker=yf_ticker)
    assert len(headlines) == 2
    assert headlines[0]["title"] == "Company beats earnings"
    assert headlines[0]["publisher"] == "Reuters"


def test_fetch_headlines_handles_nested_content_shape():
    # Newer yfinance versions nest fields under a 'content' key.
    yf_ticker = _FakeYfTicker(news=[
        {"content": {"title": "Guidance raised", "provider": {"displayName": "CNBC"}}},
    ])
    headlines = fetch_headlines("ZZZ", yf_ticker=yf_ticker)
    assert len(headlines) == 1
    assert headlines[0]["title"] == "Guidance raised"
    assert headlines[0]["publisher"] == "CNBC"


def test_fetch_headlines_skips_entries_without_title():
    yf_ticker = _FakeYfTicker(news=[{"publisher": "Reuters"}, {"title": "Real headline"}])
    headlines = fetch_headlines("ZZZ", yf_ticker=yf_ticker)
    assert len(headlines) == 1
    assert headlines[0]["title"] == "Real headline"


def test_fetch_headlines_respects_max_headlines():
    yf_ticker = _FakeYfTicker(news=[{"title": f"Headline {i}"} for i in range(10)])
    headlines = fetch_headlines("ZZZ", yf_ticker=yf_ticker, max_headlines=3)
    assert len(headlines) == 3


def test_fetch_headlines_handles_no_news():
    yf_ticker = _FakeYfTicker(news=None)
    assert fetch_headlines("ZZZ", yf_ticker=yf_ticker) == []


class _FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, result):
        self._result = result
        self.last_call_kwargs = None

    def parse(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeParsedResponse(self._result)


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)


def test_score_ticker_news_returns_validated_news_score():
    fake_score = NewsScore(
        sentiment=0.6, event_type="earnings", confidence=0.8,
        horizon="short_term", rationale="Beat estimates.",
    )
    client = _FakeClient(fake_score)
    result = score_ticker_news("ZZZ", [{"title": "Beat earnings", "publisher": "Reuters"}], client=client)
    assert isinstance(result, NewsScore)
    assert result.sentiment == 0.6
    assert result.event_type == "earnings"


def test_score_ticker_news_passes_structured_output_format():
    fake_score = NewsScore(sentiment=0.0, event_type="routine", confidence=0.5,
                            horizon="unclear", rationale="Nothing notable.")
    client = _FakeClient(fake_score)
    score_ticker_news("ZZZ", [{"title": "Routine update"}], client=client)
    assert client.messages.last_call_kwargs["output_format"] is NewsScore
    assert "ZZZ" in client.messages.last_call_kwargs["messages"][0]["content"]


def test_build_news_scores_skips_tickers_with_no_headlines():
    fake_score = NewsScore(sentiment=0.3, event_type="product", confidence=0.6,
                            horizon="long_term", rationale="New product well received.")
    client = _FakeClient(fake_score)

    def yf_factory(ticker):
        if ticker == "HASNEWS":
            return _FakeYfTicker(news=[{"title": "Something happened"}])
        return _FakeYfTicker(news=[])

    scores = build_news_scores(["HASNEWS", "NONEWS"], client=client, yf_ticker_factory=yf_factory)
    assert "HASNEWS" in scores
    assert "NONEWS" not in scores


def test_build_news_scores_stops_early_on_credential_failure():
    class _FailingMessages:
        def parse(self, **kwargs):
            raise RuntimeError("AuthenticationError: invalid credential")

    class _FailingClient:
        messages = _FailingMessages()

    def yf_factory(ticker):
        return _FakeYfTicker(news=[{"title": "Some headline"}])

    scores = build_news_scores(["A", "B", "C"], client=_FailingClient(), yf_ticker_factory=yf_factory)
    assert scores == {}


def test_log_news_scores_writes_one_jsonl_line_per_ticker(tmp_path):
    log_path = tmp_path / "news_log.jsonl"
    scores = {
        "AAA": NewsScore(sentiment=0.5, event_type="earnings", confidence=0.7,
                          horizon="short_term", rationale="Beat."),
        "BBB": NewsScore(sentiment=-0.2, event_type="legal", confidence=0.4,
                          horizon="long_term", rationale="Lawsuit filed."),
    }
    log_news_scores(scores, path=str(log_path))

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    tickers = {p["ticker"] for p in parsed}
    assert tickers == {"AAA", "BBB"}
    for p in parsed:
        assert "date" in p
        assert "sentiment" in p


def test_log_news_scores_appends_without_overwriting(tmp_path):
    log_path = tmp_path / "news_log.jsonl"
    score = NewsScore(sentiment=0.1, event_type="routine", confidence=0.5,
                       horizon="unclear", rationale="x")
    log_news_scores({"AAA": score}, path=str(log_path))
    log_news_scores({"BBB": score}, path=str(log_path))

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
