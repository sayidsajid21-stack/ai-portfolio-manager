import os

import pytest

from alpaca_execution import (
    is_configured, get_account, get_positions, positions_to_dollar_values,
    compute_rebalance_orders, place_order, rebalance_portfolio, MIN_ORDER_DOLLARS,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class _FakeSession:
    """Routes .get()/.post() to canned responses keyed by path suffix."""

    def __init__(self, account=None, positions=None, order_result=None, fail_order=False):
        self.account = account or {"equity": "100000"}
        self.positions = positions if positions is not None else []
        self.order_result = order_result or {"id": "order-123", "status": "accepted"}
        self.fail_order = fail_order
        self.posted_payloads = []

    def get(self, url, headers=None, timeout=None):
        if url.endswith("/account"):
            return _FakeResponse(self.account)
        if url.endswith("/positions"):
            return _FakeResponse(self.positions)
        raise AssertionError(f"Unexpected GET {url}")

    def post(self, url, json=None, headers=None, timeout=None):
        self.posted_payloads.append(json)
        if self.fail_order:
            return _FakeResponse({"message": "insufficient buying power"}, status=422)
        return _FakeResponse(self.order_result)


@pytest.fixture(autouse=True)
def _clear_alpaca_env(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)


def test_is_configured_false_when_unset():
    assert is_configured() is False


def test_is_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    assert is_configured() is True


def test_is_configured_false_when_only_one_set(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    assert is_configured() is False


def test_get_account_returns_parsed_json():
    session = _FakeSession(account={"equity": "50000", "cash": "1000"})
    account = get_account(session=session)
    assert account["equity"] == "50000"


def test_get_positions_returns_empty_list_not_error():
    session = _FakeSession(positions=[])
    assert get_positions(session=session) == []


def test_positions_to_dollar_values():
    positions = [
        {"symbol": "AAPL", "market_value": "1234.56"},
        {"symbol": "AGG", "market_value": "500.00"},
    ]
    result = positions_to_dollar_values(positions)
    assert result == {"AAPL": 1234.56, "AGG": 500.00}


def test_compute_rebalance_orders_buys_to_reach_target():
    target_weights = {"AAPL": 0.10}
    orders = compute_rebalance_orders(target_weights, account_equity=100_000, current_dollar_values={})
    assert len(orders) == 1
    assert orders[0] == {"symbol": "AAPL", "side": "buy", "notional": 10000.0}


def test_compute_rebalance_orders_sells_when_overweight():
    target_weights = {"AAPL": 0.05}
    current = {"AAPL": 10_000.0}
    orders = compute_rebalance_orders(target_weights, account_equity=100_000, current_dollar_values=current)
    assert len(orders) == 1
    assert orders[0]["side"] == "sell"
    assert orders[0]["notional"] == pytest.approx(5000.0)


def test_compute_rebalance_orders_sells_entirely_out_of_position_not_in_target():
    # A position Alpaca holds that isn't in target_weights at all should
    # be fully sold, not silently ignored.
    orders = compute_rebalance_orders({}, account_equity=100_000, current_dollar_values={"XOM": 2000.0})
    assert len(orders) == 1
    assert orders[0] == {"symbol": "XOM", "side": "sell", "notional": 2000.0}


def test_compute_rebalance_orders_skips_tiny_diffs():
    target_weights = {"AAPL": 0.10001}  # diff of a few cents from current
    current = {"AAPL": 10_000.0}
    orders = compute_rebalance_orders(target_weights, account_equity=100_000, current_dollar_values=current,
                                       min_order_dollars=MIN_ORDER_DOLLARS)
    assert orders == []


def test_compute_rebalance_orders_does_not_skip_diffs_right_at_threshold():
    target_weights = {"AAPL": 0.102}  # diff = $200, above MIN_ORDER_DOLLARS
    current = {"AAPL": 10_000.0}
    orders = compute_rebalance_orders(target_weights, account_equity=100_000, current_dollar_values=current,
                                       min_order_dollars=10.0)
    assert len(orders) == 1


def test_place_order_sends_notional_payload():
    session = _FakeSession()
    order = {"symbol": "AAPL", "side": "buy", "notional": 123.45}
    result = place_order(order, session=session)
    assert result["status"] == "accepted"
    payload = session.posted_payloads[0]
    assert payload["symbol"] == "AAPL"
    assert payload["side"] == "buy"
    assert payload["notional"] == "123.45"
    assert payload["type"] == "market"


def test_rebalance_portfolio_dry_run_does_not_submit():
    session = _FakeSession(account={"equity": "100000"}, positions=[])
    orders = rebalance_portfolio({"AAPL": 0.10}, session=session, dry_run=True)
    assert len(orders) == 1
    assert orders[0]["submitted"] is False
    assert session.posted_payloads == []  # nothing was actually posted


def test_rebalance_portfolio_live_submits_orders():
    session = _FakeSession(account={"equity": "100000"}, positions=[])
    orders = rebalance_portfolio({"AAPL": 0.10}, session=session, dry_run=False)
    assert len(orders) == 1
    assert orders[0]["submitted"] is True
    assert orders[0]["order_id"] == "order-123"
    assert len(session.posted_payloads) == 1


def test_rebalance_portfolio_handles_per_order_failure_without_aborting():
    session = _FakeSession(account={"equity": "100000"}, positions=[], fail_order=True)
    orders = rebalance_portfolio({"AAPL": 0.10, "MSFT": 0.10}, session=session, dry_run=False)
    assert len(orders) == 2
    for order in orders:
        assert order["submitted"] is False
        assert "error" in order


def test_rebalance_portfolio_reconciles_existing_positions():
    session = _FakeSession(
        account={"equity": "100000"},
        positions=[{"symbol": "AAPL", "market_value": "8000"}],
    )
    # Target is 10% ($10,000); current is $8,000 -> should buy $2,000 more.
    orders = rebalance_portfolio({"AAPL": 0.10}, session=session, dry_run=True)
    assert len(orders) == 1
    assert orders[0]["side"] == "buy"
    assert orders[0]["notional"] == pytest.approx(2000.0)
