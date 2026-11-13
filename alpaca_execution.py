"""
Paper-trading execution via Alpaca's Trading API (Phase 6).

Uses raw REST calls (via `requests`) against Alpaca's paper-trading
endpoint rather than a dedicated SDK, to avoid adding a new dependency
beyond what's already in requirements.txt. All functions take an
injectable `session` for testing without real network access — tests
use a fake session, never the real API.

Safety posture:
  - Defaults to PAPER trading (https://paper-api.alpaca.markets) — never
    hardcode or default to the live-trading endpoint.
  - Defaults to dry_run=True: rebalance_portfolio() computes and prints
    the orders it *would* place but does not submit them unless
    dry_run=False is passed explicitly. rebalance.py's wiring reads this
    from the ALPACA_DRY_RUN environment variable, defaulting to "true".
  - Orders below MIN_ORDER_DOLLARS are skipped — this is paper trading
    for a long-term-investing strategy, not a reason to submit dozens of
    $3 orders every run.
  - Target weights passed in should already have gone through
    portfolio_lib.apply_bond_floor and portfolio_lib.cap_single_asset_weight
    — this module does not re-derive or re-check those rules; it trusts
    the weights it's given and only translates them into orders.
"""
import os

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2"
MIN_ORDER_DOLLARS = 10.0


def _headers():
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }


def _base_url():
    return os.environ.get("ALPACA_BASE_URL", DEFAULT_BASE_URL)


def is_configured() -> bool:
    """True if both Alpaca credentials are set — rebalance.py uses this
    to decide whether to attempt execution at all."""
    return bool(os.environ.get("ALPACA_API_KEY")) and bool(os.environ.get("ALPACA_SECRET_KEY"))


def get_account(session=None) -> dict:
    """Fetch account info (equity, cash, buying_power, ...)."""
    import requests
    session = session or requests
    response = session.get(f"{_base_url()}/account", headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def get_positions(session=None) -> list:
    """Fetch current positions. Returns [] if there are none (Alpaca
    returns 200 + [] for an empty portfolio, not a 404)."""
    import requests
    session = session or requests
    response = session.get(f"{_base_url()}/positions", headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def positions_to_dollar_values(positions: list) -> dict:
    """{symbol: market_value} from Alpaca's position objects."""
    return {p["symbol"]: float(p["market_value"]) for p in positions}


def compute_rebalance_orders(target_weights: dict, account_equity: float,
                              current_dollar_values: dict,
                              min_order_dollars: float = MIN_ORDER_DOLLARS) -> list:
    """
    Diff target dollar allocations against current positions and return
    the list of orders needed to close the gap, skipping anything below
    `min_order_dollars` (not worth the noise/costs for a long-term
    strategy). Each order is {symbol, side, notional} where `notional`
    is a positive dollar amount.
    """
    orders = []
    all_symbols = set(target_weights) | set(current_dollar_values)

    for symbol in sorted(all_symbols):
        target_dollars = target_weights.get(symbol, 0.0) * account_equity
        current_dollars = current_dollar_values.get(symbol, 0.0)
        diff = target_dollars - current_dollars

        if abs(diff) < min_order_dollars:
            continue

        orders.append({
            "symbol": symbol,
            "side": "buy" if diff > 0 else "sell",
            "notional": round(abs(diff), 2),
        })

    return orders


def place_order(order: dict, session=None) -> dict:
    """Submit a single market, day, notional (dollar-denominated) order."""
    import requests
    session = session or requests
    payload = {
        "symbol": order["symbol"],
        "notional": str(order["notional"]),
        "side": order["side"],
        "type": "market",
        "time_in_force": "day",
    }
    response = session.post(f"{_base_url()}/orders", json=payload, headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def rebalance_portfolio(target_weights: dict, session=None, dry_run: bool = True,
                         min_order_dollars: float = MIN_ORDER_DOLLARS) -> list:
    """
    Orchestrates a full rebalance: fetch account equity and current
    positions, compute the orders needed to reach `target_weights`, and
    (unless dry_run) submit them.

    Returns the list of orders that were computed — whether or not they
    were actually submitted, so the caller can log/print them either way.
    Each returned order additionally carries 'submitted': bool and, if
    submitted, 'order_id' and 'status' from Alpaca's response; on a
    per-order submission failure, 'submitted': False and 'error': str
    instead of raising, so one bad order doesn't abort the rest of the
    rebalance.
    """
    account = get_account(session=session)
    equity = float(account["equity"])

    positions = get_positions(session=session)
    current_dollar_values = positions_to_dollar_values(positions)

    orders = compute_rebalance_orders(target_weights, equity, current_dollar_values, min_order_dollars)

    for order in orders:
        if dry_run:
            order["submitted"] = False
            order["note"] = "dry run — not submitted"
            continue
        try:
            result = place_order(order, session=session)
            order["submitted"] = True
            order["order_id"] = result.get("id")
            order["status"] = result.get("status")
        except Exception as e:
            order["submitted"] = False
            order["error"] = str(e)

    return orders


def print_order_report(orders: list, dry_run: bool):
    if not orders:
        print("   No rebalancing orders needed — current positions are already close enough to target.")
        return

    mode = "DRY RUN (not submitted)" if dry_run else "LIVE (paper account)"
    print(f"   Rebalancing orders [{mode}]:")
    for order in orders:
        status = order.get("status") or order.get("note") or order.get("error") or "pending"
        print(f"     {order['side']:<4} ${order['notional']:>10,.2f}  {order['symbol']:<6} | {status}")
