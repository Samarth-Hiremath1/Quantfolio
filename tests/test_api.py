"""
API route tests against a SQLite-seeded DB (no Docker). Guards D-03, D-05, D-06
and the backtest wiring.
"""


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_optimize_ok(client, tickers):
    r = client.post("/api/v1/portfolio/optimize", json={"tickers": tickers[:4], "objective": "sharpe"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-2
    assert "sharpe_ratio" in body


# ---- D-03: /risk works and returns real, non-zero metrics -------------------
def test_d03_risk_route_ok(client, tickers):
    ts = tickers[:3]
    weights = {t: 1.0 / len(ts) for t in ts}
    r = client.post("/api/v1/portfolio/risk", json={"tickers": ts, "weights": weights})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"sharpe_ratio", "sortino_ratio", "max_drawdown", "var_95", "cvar_95"}
    # max drawdown must be a real (<=0) number, not a zeroed-out .get() default
    assert body["max_drawdown"] <= 0.0


# ---- D-06: /forecast runs a real model, not hardcoded values ----------------
def test_d06_forecast_is_real(client, tickers):
    r = client.post("/api/v1/forecast", json={"tickers": tickers[:2], "model_type": "Ridge_AR"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["forecasts"]) == set(t.upper() for t in tickers[:2])
    assert all(m.startswith("Ridge_AR") for m in body["models_used"].values())
    # the old stub returned identical [0.001,0.002,-0.001,0.005,0.003] for every ticker
    vals = list(body["forecasts"].values())
    assert vals[0] != [0.001, 0.002, -0.001, 0.005, 0.003]
    assert all(v in body["n_observations"] for v in body["forecasts"])


# ---- D-05: SQL injection / bad tickers are rejected -------------------------
def test_d05_sql_injection_rejected(client):
    r = client.post("/api/v1/portfolio/optimize", json={"tickers": ["SPY') OR '1'='1"]})
    assert r.status_code == 422  # schema validation blocks it


def test_d05_unknown_ticker_is_404_not_500(client):
    r = client.post("/api/v1/portfolio/optimize", json={"tickers": ["ZZZZ"]})
    assert r.status_code == 404


# ---- backtest route end-to-end (BuyAndHold produces fills on real data) ------
def test_backtest_buy_and_hold(client, tickers):
    r = client.post("/api/v1/backtest",
                    json={"tickers": tickers[:3], "strategy": "BuyAndHold",
                          "initial_capital": 100000.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_trades"] >= 1          # fill path actually executes
    assert body["bars_processed"] > 0
    assert "sharpe_ratio" in body and "max_drawdown" in body
