"""
Monthly report generation (Phase 7).

Builds a single readable Markdown report covering what the original
project plan asked for: positions, changes from last month, any
news-driven adjustments, performance against the baselines, and the
orders that were (or would be) placed. Pure string-building logic, no
network calls or file I/O here — that's rebalance.py's job (reading/
writing last_recommendation.json and reports/*.md). This separation is
what makes it unit-testable without running the whole monthly pipeline.
"""


def format_weights_table(weights: dict, all_tickers, equity_tickers, bond_tickers) -> str:
    lines = ["| Asset | Type | Weight |", "|---|---|---|"]
    for ticker in all_tickers:
        asset_type = "Equity" if ticker in equity_tickers else "Bond"
        lines.append(f"| {ticker} | {asset_type} | {weights.get(ticker, 0.0):.2%} |")

    total_equity = sum(weights.get(t, 0.0) for t in equity_tickers)
    total_bond = sum(weights.get(t, 0.0) for t in bond_tickers)
    lines.append(f"| **Total equity** | | **{total_equity:.2%}** |")
    lines.append(f"| **Total bonds** | | **{total_bond:.2%}** |")
    return "\n".join(lines)


def format_weight_changes(current: dict, previous: dict, min_change: float = 0.005) -> str:
    """
    A table of tickers whose weight changed by more than `min_change`
    (0.5 percentage points by default) since the previous recommendation
    — small rounding-level drift isn't worth reporting every month.
    `previous` may be None (no prior recommendation exists yet, e.g. the
    very first run).
    """
    if previous is None:
        return "_No previous recommendation to compare against (this is the first report)._"

    all_tickers = sorted(set(current) | set(previous))
    rows = []
    for ticker in all_tickers:
        before = previous.get(ticker, 0.0)
        after = current.get(ticker, 0.0)
        change = after - before
        if abs(change) >= min_change:
            arrow = "▲" if change > 0 else "▼"
            rows.append(f"| {ticker} | {before:.2%} | {after:.2%} | {arrow} {change:+.2%} |")

    if not rows:
        return "_No material changes (all weights moved by less than 0.5 percentage points)._"

    return "\n".join(["| Asset | Last month | This month | Change |", "|---|---|---|---|"] + rows)


def format_news_section(news_scores: dict) -> str:
    """`news_scores` maps ticker -> an object with sentiment/event_type/
    confidence/horizon/rationale attributes (news_sentiment.NewsScore),
    or is empty/None if the overlay didn't run or found nothing."""
    if not news_scores:
        return "_No news adjustments this month (no headlines found, or no Claude credential configured)._"

    rows = []
    for ticker, score in news_scores.items():
        rows.append(
            f"| {ticker} | {score.sentiment:+.2f} | {score.event_type} | "
            f"{score.confidence:.0%} | {score.rationale} |"
        )
    return "\n".join(["| Asset | Sentiment | Event type | Confidence | Rationale |", "|---|---|---|---|---|"] + rows)


def format_performance_section(new_model_return: float, baseline_return: float, old_model_return) -> str:
    lines = [
        "| | Holdout return |",
        "|---|---|",
        f"| This month's model | {new_model_return:.2%} |",
        f"| Equal-weight baseline | {baseline_return:.2%} |",
    ]
    if old_model_return is not None:
        lines.append(f"| Previously saved model | {old_model_return:.2%} |")

    beat_baseline = new_model_return > baseline_return
    verdict = (
        "✅ This month's model beat the equal-weight baseline on held-out data."
        if beat_baseline else
        "⚠️ This month's model did NOT beat the equal-weight baseline — the previous model was kept."
    )
    return "\n".join(lines) + f"\n\n{verdict}"


def format_orders_section(orders: list, dry_run: bool, alpaca_configured: bool) -> str:
    if not alpaca_configured:
        return "_Alpaca execution not configured (ALPACA_API_KEY/ALPACA_SECRET_KEY unset) — recommendation only._"
    if not orders:
        return "_No rebalancing orders needed — current positions are already close to target._"

    mode = "**DRY RUN — not submitted**" if dry_run else "**LIVE — submitted to paper account**"
    rows = []
    for order in orders:
        status = order.get("status") or order.get("note") or order.get("error") or "pending"
        rows.append(f"| {order['side']} | {order['symbol']} | ${order['notional']:,.2f} | {status} |")

    return mode + "\n\n" + "\n".join(["| Side | Asset | Amount | Status |", "|---|---|---|---|"] + rows)


def build_report(report_date, current_weights: dict, previous_weights, news_scores: dict,
                  new_model_return: float, baseline_return: float, old_model_return,
                  orders: list, dry_run: bool, alpaca_configured: bool,
                  all_tickers, equity_tickers, bond_tickers) -> str:
    """Assemble the full Markdown report for one monthly run."""
    sections = [
        f"# AI Portfolio Manager — Monthly Report ({report_date})",
        "",
        "## Recommended allocation",
        format_weights_table(current_weights, all_tickers, equity_tickers, bond_tickers),
        "",
        "## Changes since last month",
        format_weight_changes(current_weights, previous_weights),
        "",
        "## News sentiment adjustments",
        format_news_section(news_scores),
        "",
        "## Performance vs. baseline (held-out data)",
        format_performance_section(new_model_return, baseline_return, old_model_return),
        "",
        "## Paper trading orders",
        format_orders_section(orders, dry_run, alpaca_configured),
        "",
    ]
    return "\n".join(sections)
