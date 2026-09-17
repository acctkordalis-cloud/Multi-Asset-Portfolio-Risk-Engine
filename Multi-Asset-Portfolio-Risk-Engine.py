


# =============================================================================
#  MULTI-ASSET PORTFOLIO ENGINE
# =============================================================================
# Initial capital: EUR 150,000
#
# ACTUAL CAPITAL SLEEVES:
#   Equities:              EUR 60,000
#   Bonds:                 EUR 40,000
#   Derivatives Reserve:   EUR 20,000
#   General Cash:          EUR 30,000
#   TOTAL:                 EUR 150,000
#
# IMPORTANT:
# - The EUR 20k derivatives sleeve is real capital, initially held as an
#   earmarked cash reserve and used to fund option/CDS cash flows.
# - General cash remains separate and receives dividends, bond coupons and
#   cash interest.
# - This avoids the previous inconsistency where a "EUR 20k derivatives budget"
#   was mentioned but not represented as an asset in initial NAV.
#
# MARKET DATA:
# - Equities / SPY / EURUSD / dividends: yfinance
# - Bonds: duration + convexity model using ^TNX as a rate proxy
# - CDS: user CSV if supplied, otherwise explicitly-labelled HYG stress proxy
#
# MAJOR CORRECTIONS IN THIS VERSION:
# 1) Unified time-series NAV for every sleeve
# 2) Initial NAV reconciles to EUR 150,000
# 3) Missing-data report before valuation filling
# 4) Actual historical dividends
# 5) Historical cash ledger
# 6) Separate derivative reserve
# 7) Rolling portfolio-level SPY put hedge
# 8) Option stress uses Black-Scholes repricing
# 9) Volatility stress uses MULTIPLIERS, not +50 percentage points
# 10) Delta / Gamma / hedge-ratio diagnostics
# 11) Bonds start at exactly EUR 40,000
# 12) CDS is fair at inception: contractual coupon = inception spread proxy
# 13) CDS-only stress test
# 14) Full multi-asset stress test
# 15) Total-portfolio VaR / ES / Sharpe / Sortino / drawdown
# 16) Equity benchmark + blended total benchmark
# 17) CVA / counterparty exposure
#
# Install once:
# %pip install yfinance pandas numpy scipy matplotlib
# =============================================================================

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

# =============================================================================
# 1. SETTINGS
# =============================================================================

START_DATE = "2020-01-01"
DOWNLOAD_START = "2019-01-01"
TRADING_DAYS = 252

INITIAL_CAPITAL = 150_000.0
EQUITY_ALLOCATION = 60_000.0
BOND_ALLOCATION = 40_000.0
DERIVATIVE_RESERVE_START = 20_000.0
GENERAL_CASH_START = 30_000.0

RISK_FREE_RATE = 0.035
GENERAL_CASH_YIELD = 0.035
DERIVATIVE_RESERVE_YIELD = 0.030

VAR_CONFIDENCE = 0.99
MIN_GENERAL_CASH_PCT = 0.10

# Equity hedge
TARGET_HEDGE_RATIO = 0.50
PUT_MONEYNESS = 0.95
PUT_TENOR_YEARS = 0.50
HEDGE_REBALANCE_MONTHS = 3
OPTION_MULTIPLIER = 100
MIN_VOL = 0.12
MAX_VOL = 1.50

# CDS / CCR
CDS_NOTIONAL = 100_000.0
CDS_RECOVERY = 0.40
CDS_MATURITY = 5
COUNTERPARTY_PD = 0.02
COUNTERPARTY_RECOVERY = 0.40

USE_ACTUAL_CDS_CSV = False
CDS_CSV_PATH = "cds_spreads.csv"

RISK_LIMITS = {
    "max_drawdown": -0.15,
    "var_pct": 0.05,
    "single_stock_weight": 0.15,
    "minimum_general_cash_pct": MIN_GENERAL_CASH_PCT,
}

OUTPUT_DIR = Path("portfolio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 2. FIXED EQUITY BOOK
# =============================================================================
# NOTE:
# This is a fixed exogenous portfolio chosen by the user.
# Therefore the historical result is a PORTFOLIO BACKTEST, not proof of
# ex-ante stock-selection skill. A true walk-forward selection engine would
# require a pre-defined historical investable universe and ranking rules.

stock_portfolio_eur = {
    "MSFT": 7000,
    "GOOGL": 7000,
    "AMZN": 6000,
    "JPM": 6000,
    "V": 5000,
    "ASML": 5000,
    "XOM": 5000,
    "JNJ": 5000,
    "PG": 4000,
    "CAT": 5000,
    "META": 5000,
}

if not np.isclose(sum(stock_portfolio_eur.values()), EQUITY_ALLOCATION):
    raise ValueError("Equity allocations must sum to EUR 60,000.")

TICKERS = list(stock_portfolio_eur.keys())

# =============================================================================
# 3. HELPERS
# =============================================================================

def download_close(tickers, start=DOWNLOAD_START):
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.sort_index()

def bs_put(S, K, T, r, sigma):
    if S <= 0 or K <= 0:
        return 0.0

    if T <= 0:
        return float(max(K - S, 0.0))

    sigma = float(np.clip(sigma, 1e-6, MAX_VOL))

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return float(
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )

def put_delta(S, K, T, r, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0

    sigma = float(np.clip(sigma, 1e-6, MAX_VOL))

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return float(norm.cdf(d1) - 1.0)

def option_gamma(S, K, T, r, sigma):
    if T <= 0 or S <= 0:
        return 0.0

    sigma = float(np.clip(sigma, 1e-6, MAX_VOL))

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return float(
        norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )

def cds_value(
    notional,
    coupon_bps,
    market_spread_bps,
    maturity,
    r,
    recovery=0.40,
    freq=4,
):
    market_spread_bps = max(float(market_spread_bps), 0.01)
    coupon_bps = max(float(coupon_bps), 0.01)

    hazard = (
        market_spread_bps / 10_000
    ) / (1 - recovery)

    dt = 1 / freq
    times = np.arange(dt, maturity + 1e-12, dt)

    premium_leg = 0.0
    protection_leg = 0.0
    prev_survival = 1.0
    coupon = coupon_bps / 10_000

    for t in times:
        survival = np.exp(-hazard * t)
        discount = np.exp(-r * t)

        premium_leg += (
            notional
            * coupon
            * dt
            * survival
            * discount
        )

        default_prob = prev_survival - survival

        protection_leg += (
            notional
            * (1 - recovery)
            * default_prob
            * discount
        )

        prev_survival = survival

    return float(protection_leg - premium_leg)

def historical_var_es(returns, nav, confidence=0.99):
    clean = returns.dropna()

    if clean.empty:
        return np.nan, np.nan

    q = np.quantile(clean, 1 - confidence)
    var = abs(q) * nav

    tail = clean[clean <= q]
    es = abs(tail.mean()) * nav if len(tail) else np.nan

    return float(var), float(es)

# =============================================================================
# 4. RAW MARKET DATA
# =============================================================================

raw_equity_usd = download_close(TICKERS)
raw_spy_usd = download_close(["SPY"])["SPY"]
raw_fx = download_close(["EURUSD=X"])["EURUSD=X"]
raw_tnx = download_close(["^TNX"])["^TNX"]
raw_hyg = download_close(["HYG"])["HYG"]
raw_sp500 = download_close(["^GSPC"])["^GSPC"]

strategy_index = raw_equity_usd.loc[START_DATE:].index

equity_raw = raw_equity_usd.reindex(strategy_index)
spy = raw_spy_usd.reindex(strategy_index).ffill()
eurusd = raw_fx.reindex(strategy_index).ffill()
tnx = raw_tnx.reindex(strategy_index).ffill()
hyg = raw_hyg.reindex(strategy_index).ffill()
sp500 = raw_sp500.reindex(strategy_index).ffill()

# -----------------------------------------------------------------------------
# Missing observations report ONLY — does not alter raw data
# -----------------------------------------------------------------------------

missing_report = pd.DataFrame({
    "Total Rows": len(equity_raw),
    "Available Observations": equity_raw.notna().sum(),
    "Missing Observations": equity_raw.isna().sum(),
    "Missing %": equity_raw.isna().mean() * 100,
})

print("\n" + "=" * 92)
print("MISSING OBSERVATIONS PER STOCK")
print("=" * 92)
print(missing_report)

missing_report.to_csv(
    OUTPUT_DIR / "missing_observations_per_stock.csv"
)

# Valuation-only forward fill.
# Risk return matrices below still use raw prices with fill_method=None.
equity_val_prices = equity_raw.ffill()

first_valid_date = (
    equity_val_prices.dropna(how="any")
    .index[0]
)

# =============================================================================
# 5. FX-AWARE EQUITY BOOK
# =============================================================================

initial_fx = float(eurusd.loc[first_valid_date])
initial_prices = equity_val_prices.loc[first_valid_date]

shares = pd.Series({
    ticker: (
        stock_portfolio_eur[ticker]
        * initial_fx
        / initial_prices[ticker]
    )
    for ticker in TICKERS
})

equity_values_eur = pd.DataFrame(
    index=strategy_index
)

for ticker in TICKERS:
    equity_values_eur[ticker] = (
        equity_val_prices[ticker]
        * shares[ticker]
        / eurusd
    )

equity_nav = (
    equity_values_eur
    .sum(axis=1, min_count=1)
    .loc[first_valid_date:]
)

equity_weights = (
    equity_values_eur
    .div(equity_nav, axis=0)
)

# Raw returns: no automatic forward fill.
equity_returns_raw = (
    raw_equity_usd
    .loc[equity_nav.index, TICKERS]
    .pct_change(fill_method=None)
)

pairwise_cov = (
    equity_returns_raw
    .cov(min_periods=60)
    * TRADING_DAYS
)

pairwise_corr = (
    equity_returns_raw
    .corr(min_periods=60)
)

pairwise_cov.to_csv(
    OUTPUT_DIR / "pairwise_covariance.csv"
)

pairwise_corr.to_csv(
    OUTPUT_DIR / "pairwise_correlation.csv"
)

# =============================================================================
# 6. ACTUAL DIVIDEND CASH FLOWS
# =============================================================================

dividend_cash = pd.Series(
    0.0,
    index=equity_nav.index
)

dividend_rows = []

for ticker in TICKERS:
    try:
        divs = yf.Ticker(ticker).dividends.copy()

        if len(divs) == 0:
            continue

        if getattr(divs.index, "tz", None) is not None:
            divs.index = divs.index.tz_localize(None)

        divs = divs[
            (divs.index >= pd.Timestamp(START_DATE))
            & (divs.index <= equity_nav.index.max())
        ]

        for dt, dps in divs.items():
            pos = equity_nav.index.searchsorted(dt)

            if pos >= len(equity_nav.index):
                continue

            pay_date = equity_nav.index[pos]

            cash_eur = (
                float(dps)
                * float(shares[ticker])
                / float(eurusd.loc[pay_date])
            )

            dividend_cash.loc[pay_date] += cash_eur

            dividend_rows.append({
                "Date": pay_date,
                "Ticker": ticker,
                "Dividend per Share USD": float(dps),
                "Shares": float(shares[ticker]),
                "Cash Dividend EUR": cash_eur,
            })

    except Exception as exc:
        print(
            f"Dividend warning for {ticker}: {exc}"
        )

pd.DataFrame(dividend_rows).to_csv(
    OUTPUT_DIR / "dividend_cashflows.csv",
    index=False
)

# =============================================================================
# 7. BOND BOOK — EXACT EUR 40,000 INITIAL MARKET VALUE
# =============================================================================

bonds = pd.DataFrame({
    "Issuer": [
        "Bond A",
        "Bond B",
        "Bond C",
        "Bond D"
    ],
    "Target Initial MV": [
        10_000,
        10_000,
        10_000,
        10_000
    ],
    "Coupon": [
        0.042,
        0.048,
        0.051,
        0.044
    ],
    "Initial Price": [
        99.0,
        101.5,
        97.5,
        100.2
    ],
    "Modified Duration": [
        3.5,
        5.2,
        6.0,
        4.1
    ],
    "Convexity": [
        18.0,
        30.0,
        39.0,
        23.0
    ],
})

# Face value is solved so each initial market value is exactly EUR 10k.
bonds["Face Value"] = (
    bonds["Target Initial MV"]
    / (bonds["Initial Price"] / 100)
)

if not np.isclose(
    bonds["Target Initial MV"].sum(),
    BOND_ALLOCATION
):
    raise ValueError(
        "Bond allocation does not equal EUR 40,000."
    )

proxy_yield = tnx / 100.0

dy = (
    proxy_yield
    .diff()
    .reindex(equity_nav.index)
    .fillna(0.0)
)

bond_value = pd.Series(
    0.0,
    index=equity_nav.index
)

bond_coupon_cash = pd.Series(
    0.0,
    index=equity_nav.index
)

bond_detail = {}

for _, row in bonds.iterrows():

    initial_mv = float(
        row["Target Initial MV"]
    )

    daily_price_return = (
        -row["Modified Duration"] * dy
        + 0.5
        * row["Convexity"]
        * dy**2
    )

    modeled = (
        initial_mv
        * (1 + daily_price_return).cumprod()
    )

    bond_detail[row["Issuer"]] = modeled
    bond_value += modeled

    annual_coupon = (
        row["Face Value"]
        * row["Coupon"]
    )

    for year in range(
        first_valid_date.year + 1,
        equity_nav.index.max().year + 1
    ):
        target = pd.Timestamp(
            year=year,
            month=first_valid_date.month,
            day=min(first_valid_date.day, 28)
        )

        pos = equity_nav.index.searchsorted(
            target
        )

        if pos < len(equity_nav.index):
            bond_coupon_cash.iloc[pos] += annual_coupon

bond_detail = pd.DataFrame(
    bond_detail
)

bond_detail.to_csv(
    OUTPUT_DIR / "bond_modeled_values.csv"
)

# =============================================================================
# 8. CDS HISTORICAL SERIES — FAIR AT INCEPTION
# =============================================================================

if (
    USE_ACTUAL_CDS_CSV
    and Path(CDS_CSV_PATH).exists()
):
    cds_csv = (
        pd.read_csv(
            CDS_CSV_PATH,
            parse_dates=["Date"]
        )
        .set_index("Date")
    )

    cds_spread = (
        cds_csv["Spread_bps"]
        .reindex(equity_nav.index)
        .ffill()
        .bfill()
    )

    cds_source = "Actual CDS CSV"

else:
    # Explicit proxy only, NOT actual CDS history.
    hyg_aligned = (
        hyg
        .reindex(equity_nav.index)
        .ffill()
    )

    hyg_peak = hyg_aligned.cummax()

    hyg_drawdown = (
        1 - hyg_aligned / hyg_peak
    ).clip(lower=0)

    cds_spread = (
        100
        + 2500 * hyg_drawdown
    ).clip(
        lower=75,
        upper=750
    )

    cds_source = "HYG credit-stress proxy"

# Contract coupon fixed at inception spread.
# This makes the CDS approximately fair-value zero at inception.
CDS_CONTRACT_COUPON_BPS = float(
    cds_spread.iloc[0]
)

raw_cds_mtm = cds_spread.apply(
    lambda spread: cds_value(
        CDS_NOTIONAL,
        CDS_CONTRACT_COUPON_BPS,
        float(spread),
        CDS_MATURITY,
        RISK_FREE_RATE,
        CDS_RECOVERY,
    )
)

# Remove tiny discretization residual so inception MTM is exactly zero.
cds_initial_residual = float(
    raw_cds_mtm.iloc[0]
)

cds_mtm = (
    raw_cds_mtm
    - cds_initial_residual
)

# CDS premium is based on the contractual coupon.
cds_premium_cash = pd.Series(
    0.0,
    index=equity_nav.index
)

quarterly_cds_premium = (
    CDS_NOTIONAL
    * CDS_CONTRACT_COUPON_BPS
    / 10_000
    / 4
)

for dt in pd.date_range(
    first_valid_date,
    equity_nav.index.max(),
    freq="QS"
):
    pos = equity_nav.index.searchsorted(dt)

    if pos < len(equity_nav.index):
        cds_premium_cash.iloc[pos] -= (
            quarterly_cds_premium
        )

# =============================================================================
# 9. ROLLING PORTFOLIO-LEVEL SPY PUT HEDGE
# =============================================================================

spy = (
    spy
    .reindex(equity_nav.index)
    .ffill()
)

spy_raw_returns = (
    raw_spy_usd
    .pct_change(fill_method=None)
)

rolling_vol = (
    spy_raw_returns
    .rolling(
        63,
        min_periods=20
    )
    .std()
    * np.sqrt(TRADING_DAYS)
)

rolling_vol = (
    rolling_vol
    .reindex(equity_nav.index)
    .ffill()
)

rolling_vol = (
    rolling_vol
    .fillna(rolling_vol.median())
    .clip(
        lower=MIN_VOL,
        upper=MAX_VOL
    )
)

equity_nav_ret = (
    equity_nav
    .pct_change(fill_method=None)
)

spy_ret = (
    spy
    .pct_change(fill_method=None)
)

rolling_beta = (
    equity_nav_ret
    .rolling(
        126,
        min_periods=60
    )
    .cov(spy_ret)
    /
    spy_ret
    .rolling(
        126,
        min_periods=60
    )
    .var()
)

rolling_beta = (
    rolling_beta
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(1.0)
    .clip(
        lower=0.25,
        upper=2.0
    )
)

option_mtm = pd.Series(
    0.0,
    index=equity_nav.index
)

option_cashflow = pd.Series(
    0.0,
    index=equity_nav.index
)

option_trade_rows = []

roll_dates = []

for dt in pd.date_range(
    first_valid_date,
    equity_nav.index.max(),
    freq=f"{HEDGE_REBALANCE_MONTHS}MS"
):
    pos = equity_nav.index.searchsorted(dt)

    if pos < len(equity_nav.index):
        actual = equity_nav.index[pos]

        if (
            not roll_dates
            or actual != roll_dates[-1]
        ):
            roll_dates.append(actual)

for i, entry_date in enumerate(
    roll_dates
):
    exit_date = (
        roll_dates[i + 1]
        if i + 1 < len(roll_dates)
        else equity_nav.index[-1]
    )

    S0 = float(
        spy.loc[entry_date]
    )

    vol0 = float(
        rolling_vol.loc[entry_date]
    )

    beta0 = float(
        rolling_beta.loc[entry_date]
    )

    nav0 = float(
        equity_nav.loc[entry_date]
    )

    fx0 = float(
        eurusd.loc[entry_date]
    )

    K = (
        S0
        * PUT_MONEYNESS
    )

    delta0 = put_delta(
        S0,
        K,
        PUT_TENOR_YEARS,
        RISK_FREE_RATE,
        vol0,
    )

    gamma0 = option_gamma(
        S0,
        K,
        PUT_TENOR_YEARS,
        RISK_FREE_RATE,
        vol0,
    )

    delta_abs = max(
        abs(delta0),
        0.05
    )

    raw_contracts = (
        nav0
        * beta0
        * TARGET_HEDGE_RATIO
        * fx0
    ) / (
        S0
        * OPTION_MULTIPLIER
        * delta_abs
    )

    contracts = max(
        1,
        math.ceil(raw_contracts)
    )

    premium_usd = bs_put(
        S0,
        K,
        PUT_TENOR_YEARS,
        RISK_FREE_RATE,
        vol0,
    )

    premium_cost_eur = (
        premium_usd
        * OPTION_MULTIPLIER
        * contracts
        / fx0
    )

    # No arbitrary contract cap based on the whole 20k sleeve.
    # Instead, if a trade cannot be funded by reserve at execution time,
    # the reserve/cash engine below will flag it transparently.

    option_cashflow.loc[
        entry_date
    ] -= premium_cost_eur

    dates = equity_nav.index[
        (equity_nav.index >= entry_date)
        & (equity_nav.index <= exit_date)
    ]

    for dt in dates:

        elapsed = (
            dt - entry_date
        ).days

        T = max(
            PUT_TENOR_YEARS
            - elapsed / 365.25,
            0.0
        )

        option_price_usd = bs_put(
            float(spy.loc[dt]),
            K,
            T,
            RISK_FREE_RATE,
            float(
                rolling_vol.loc[dt]
            ),
        )

        option_mtm.loc[dt] = (
            option_price_usd
            * OPTION_MULTIPLIER
            * contracts
            / float(
                eurusd.loc[dt]
            )
        )

    # Close old hedge at roll.
    if i + 1 < len(roll_dates):
        option_cashflow.loc[
            exit_date
        ] += option_mtm.loc[
            exit_date
        ]

    initial_delta_notional_eur = (
        contracts
        * OPTION_MULTIPLIER
        * S0
        * abs(delta0)
        / fx0
    )

    initial_hedge_ratio = (
        initial_delta_notional_eur
        / (
            nav0
            * beta0
        )
    )

    option_trade_rows.append({
        "Entry Date": entry_date,
        "Exit/Roll Date": exit_date,
        "SPY Entry": S0,
        "Strike": K,
        "Entry Vol": vol0,
        "Entry Delta": delta0,
        "Entry Gamma": gamma0,
        "Portfolio Beta": beta0,
        "Equity NAV EUR": nav0,
        "Raw Contracts": raw_contracts,
        "Contracts": contracts,
        "Premium Cost EUR": premium_cost_eur,
        "Initial Delta Notional EUR": initial_delta_notional_eur,
        "Initial Hedge Ratio": initial_hedge_ratio,
    })

option_trades = pd.DataFrame(
    option_trade_rows
)

option_trades.to_csv(
    OUTPUT_DIR / "option_hedge_trades.csv",
    index=False
)

# =============================================================================
# 10. TWO CASH ACCOUNTS
# =============================================================================
# General Cash:
#   receives dividends + bond coupons + cash yield
#
# Derivative Reserve:
#   starts at EUR 20k
#   pays/receives option hedge cash flows + CDS premiums
#   earns reserve yield
#
# If derivative reserve becomes negative, it means the hedge program
# exceeded its dedicated capital sleeve. We flag this explicitly.

general_cash = pd.Series(
    index=equity_nav.index,
    dtype=float
)

derivative_reserve = pd.Series(
    index=equity_nav.index,
    dtype=float
)

general_cash_flows = pd.DataFrame(
    index=equity_nav.index
)

general_cash_flows["Dividends"] = (
    dividend_cash
)

general_cash_flows["Bond Coupons"] = (
    bond_coupon_cash
)

derivative_cash_flows = pd.DataFrame(
    index=equity_nav.index
)

derivative_cash_flows[
    "Option Cash Flow"
] = option_cashflow

derivative_cash_flows[
    "CDS Premium"
] = cds_premium_cash

gcash = GENERAL_CASH_START
dreserve = DERIVATIVE_RESERVE_START

for i, dt in enumerate(
    equity_nav.index
):

    if i > 0:
        prev = equity_nav.index[
            i - 1
        ]

        days = (
            dt - prev
        ).days

        gcash *= (
            1
            + GENERAL_CASH_YIELD
            * days / 365.25
        )

        dreserve *= (
            1
            + DERIVATIVE_RESERVE_YIELD
            * days / 365.25
        )

    gcash += float(
        general_cash_flows
        .loc[dt]
        .sum()
    )

    dreserve += float(
        derivative_cash_flows
        .loc[dt]
        .sum()
    )

    general_cash.loc[dt] = gcash
    derivative_reserve.loc[dt] = dreserve

general_cash_flows[
    "Net General Cash Flow"
] = general_cash_flows.sum(
    axis=1
)

general_cash_flows[
    "General Cash Balance"
] = general_cash

derivative_cash_flows[
    "Net Derivative Cash Flow"
] = derivative_cash_flows.sum(
    axis=1
)

derivative_cash_flows[
    "Derivative Reserve Balance"
] = derivative_reserve

general_cash_flows.to_csv(
    OUTPUT_DIR
    / "general_cash_ledger.csv"
)

derivative_cash_flows.to_csv(
    OUTPUT_DIR
    / "derivative_reserve_ledger.csv"
)

# =============================================================================
# 11. UNIFIED TOTAL NAV
# =============================================================================

components = pd.DataFrame({
    "Equities": equity_nav,
    "Bonds": bond_value,
    "Options": option_mtm,
    "CDS": cds_mtm,
    "General Cash": general_cash,
    "Derivative Reserve": derivative_reserve,
}).dropna(
    how="any"
)

components["Total NAV"] = (
    components.sum(axis=1)
)

total_nav = components[
    "Total NAV"
]

total_returns = (
    total_nav
    .pct_change(fill_method=None)
    .dropna()
)

components.to_csv(
    OUTPUT_DIR
    / "portfolio_nav_components.csv"
)

# Check initial NAV reconciliation
initial_nav_actual = float(
    total_nav.iloc[0]
)

nav_reconciliation_error = (
    initial_nav_actual
    - INITIAL_CAPITAL
)

print("\n" + "=" * 92)
print("INITIAL NAV RECONCILIATION")
print("=" * 92)
print(
    f"Target Initial NAV: EUR {INITIAL_CAPITAL:,.2f}"
)
print(
    f"Actual Initial NAV: EUR {initial_nav_actual:,.2f}"
)
print(
    f"Difference:         EUR {nav_reconciliation_error:,.2f}"
)

# =============================================================================
# 12. TOTAL PORTFOLIO RISK
# =============================================================================

ann_return = (
    total_returns.mean()
    * TRADING_DAYS
)

ann_vol = (
    total_returns.std()
    * np.sqrt(TRADING_DAYS)
)

sharpe = (
    (
        ann_return
        - RISK_FREE_RATE
    ) / ann_vol
    if ann_vol > 0
    else np.nan
)

downside = total_returns[
    total_returns < 0
]

downside_vol = (
    downside.std()
    * np.sqrt(TRADING_DAYS)
)

sortino = (
    (
        ann_return
        - RISK_FREE_RATE
    ) / downside_vol
    if downside_vol > 0
    else np.nan
)

drawdown = (
    total_nav
    / total_nav.cummax()
    - 1
)

max_drawdown = float(
    drawdown.min()
)

current_nav = float(
    total_nav.iloc[-1]
)

var99, es99 = historical_var_es(
    total_returns,
    current_nav,
    VAR_CONFIDENCE,
)

# =============================================================================
# 13. BENCHMARK ANALYSIS
# =============================================================================

sp500 = (
    sp500
    .reindex(total_nav.index)
    .ffill()
)

sp500_ret = (
    sp500
    .pct_change(fill_method=None)
)

equity_ret = (
    equity_nav
    .reindex(total_nav.index)
    .pct_change(fill_method=None)
)

equity_cmp = pd.concat(
    [
        equity_ret.rename(
            "Portfolio"
        ),
        sp500_ret.rename(
            "Benchmark"
        )
    ],
    axis=1
).dropna()

equity_active = (
    equity_cmp["Portfolio"]
    - equity_cmp["Benchmark"]
)

equity_tracking_error = (
    equity_active.std()
    * np.sqrt(TRADING_DAYS)
)

equity_active_ann = (
    equity_cmp["Portfolio"].mean()
    - equity_cmp["Benchmark"].mean()
) * TRADING_DAYS

equity_information_ratio = (
    equity_active_ann
    / equity_tracking_error
    if equity_tracking_error > 0
    else np.nan
)

equity_beta = (
    equity_cmp["Portfolio"]
    .cov(
        equity_cmp["Benchmark"]
    )
    /
    equity_cmp["Benchmark"]
    .var()
)

portfolio_eq_ann = (
    equity_cmp["Portfolio"]
    .mean()
    * TRADING_DAYS
)

benchmark_ann = (
    equity_cmp["Benchmark"]
    .mean()
    * TRADING_DAYS
)

equity_capm_alpha = (
    portfolio_eq_ann
    - (
        RISK_FREE_RATE
        + equity_beta
        * (
            benchmark_ann
            - RISK_FREE_RATE
        )
    )
)

# Blended total benchmark
bond_ret = (
    bond_value
    .reindex(total_nav.index)
    .pct_change(fill_method=None)
    .fillna(0)
)

general_cash_daily = (
    (1 + GENERAL_CASH_YIELD)
    ** (1 / TRADING_DAYS)
    - 1
)

reserve_daily = (
    (1 + DERIVATIVE_RESERVE_YIELD)
    ** (1 / TRADING_DAYS)
    - 1
)

blended_benchmark_ret = (
    0.40
    * sp500_ret.fillna(0)
    + (
        BOND_ALLOCATION
        / INITIAL_CAPITAL
    )
    * bond_ret
    + (
        GENERAL_CASH_START
        / INITIAL_CAPITAL
    )
    * general_cash_daily
    + (
        DERIVATIVE_RESERVE_START
        / INITIAL_CAPITAL
    )
    * reserve_daily
)

blended_growth = (
    1
    + blended_benchmark_ret
).cumprod()

portfolio_growth = (
    total_nav
    / total_nav.iloc[0]
)

total_cmp = pd.concat(
    [
        total_returns.rename(
            "Portfolio"
        ),
        blended_benchmark_ret.rename(
            "Benchmark"
        )
    ],
    axis=1
).dropna()

total_active = (
    total_cmp["Portfolio"]
    - total_cmp["Benchmark"]
)

total_tracking_error = (
    total_active.std()
    * np.sqrt(TRADING_DAYS)
)

total_active_ann = (
    total_cmp["Portfolio"].mean()
    - total_cmp["Benchmark"].mean()
) * TRADING_DAYS

total_information_ratio = (
    total_active_ann
    / total_tracking_error
    if total_tracking_error > 0
    else np.nan
)

# =============================================================================
# 14. CVA / COUNTERPARTY RISK
# =============================================================================

current_cds_mtm = float(
    cds_mtm.iloc[-1]
)

current_exposure = max(
    current_cds_mtm,
    0.0
)

cva = (
    current_exposure
    * COUNTERPARTY_PD
    * (
        1
        - COUNTERPARTY_RECOVERY
    )
)

credit_adjusted_cds = (
    current_cds_mtm
    - cva
)

# =============================================================================
# 15. CDS-ONLY STRESS TEST
# =============================================================================

current_cds_spread = float(
    cds_spread.iloc[-1]
)

cds_stress_levels = [
    max(
        50,
        current_cds_spread
        - 100
    ),
    current_cds_spread,
    current_cds_spread
    + 100,
    current_cds_spread
    + 250,
    current_cds_spread
    + 500,
]

cds_rows = []

for spread in cds_stress_levels:

    stressed_raw = cds_value(
        CDS_NOTIONAL,
        CDS_CONTRACT_COUPON_BPS,
        spread,
        CDS_MATURITY,
        RISK_FREE_RATE,
        CDS_RECOVERY,
    )

    stressed_mtm = (
        stressed_raw
        - cds_initial_residual
    )

    cds_rows.append({
        "Stressed Spread bps": spread,
        "CDS MTM EUR": stressed_mtm,
        "P&L vs Current EUR": (
            stressed_mtm
            - current_cds_mtm
        ),
    })

cds_only_stress = pd.DataFrame(
    cds_rows
)

# =============================================================================
# 16. FULL MULTI-ASSET STRESS TEST
# =============================================================================
# Volatility shocks are MULTIPLIERS:
# 1.20 = volatility rises by 20%
# 2.00 = volatility doubles
#
# We do NOT cap option P&L artificially.
# Instead we report stressed delta, gamma and effective hedge ratio so that
# nonlinear over-hedging is visible rather than hidden.

stress_scenarios = [
    {
        "Scenario": "Mild Risk-Off",
        "Equity Shock": -0.05,
        "Rate Shock": 0.005,
        "Credit Spread Shock bps": 50,
        "Vol Multiplier": 1.20,
        "EURUSD Shock": 0.03,
    },
    {
        "Scenario": "Market Correction",
        "Equity Shock": -0.10,
        "Rate Shock": 0.0075,
        "Credit Spread Shock bps": 100,
        "Vol Multiplier": 1.40,
        "EURUSD Shock": 0.05,
    },
    {
        "Scenario": "Severe Equity Crash",
        "Equity Shock": -0.20,
        "Rate Shock": 0.010,
        "Credit Spread Shock bps": 200,
        "Vol Multiplier": 1.75,
        "EURUSD Shock": 0.08,
    },
    {
        "Scenario": "Credit Crisis",
        "Equity Shock": -0.12,
        "Rate Shock": 0.005,
        "Credit Spread Shock bps": 300,
        "Vol Multiplier": 1.60,
        "EURUSD Shock": 0.05,
    },
    {
        "Scenario": "Rate Shock",
        "Equity Shock": -0.05,
        "Rate Shock": 0.020,
        "Credit Spread Shock bps": 75,
        "Vol Multiplier": 1.20,
        "EURUSD Shock": 0.02,
    },
    {
        "Scenario": "Severe Crisis",
        "Equity Shock": -0.30,
        "Rate Shock": 0.015,
        "Credit Spread Shock bps": 500,
        "Vol Multiplier": 2.00,
        "EURUSD Shock": 0.10,
    },
]

current_date = (
    total_nav.index[-1]
)

current_equity = float(
    components.loc[
        current_date,
        "Equities"
    ]
)

current_bonds = float(
    components.loc[
        current_date,
        "Bonds"
    ]
)

current_options = float(
    components.loc[
        current_date,
        "Options"
    ]
)

current_general_cash = float(
    components.loc[
        current_date,
        "General Cash"
    ]
)

current_derivative_reserve = float(
    components.loc[
        current_date,
        "Derivative Reserve"
    ]
)

current_fx = float(
    eurusd.loc[current_date]
)

current_spy = float(
    spy.loc[current_date]
)

current_vol = float(
    rolling_vol.loc[
        current_date
    ]
)

last_trade = (
    option_trades.iloc[-1]
)

last_strike = float(
    last_trade["Strike"]
)

last_contracts = int(
    last_trade["Contracts"]
)

last_entry = pd.Timestamp(
    last_trade["Entry Date"]
)

current_beta = float(
    rolling_beta.loc[
        current_date
    ]
)

elapsed = (
    current_date
    - last_entry
).days

remaining_T = max(
    PUT_TENOR_YEARS
    - elapsed / 365.25,
    1 / 365.25
)

current_put_delta = put_delta(
    current_spy,
    last_strike,
    remaining_T,
    RISK_FREE_RATE,
    current_vol,
)

current_put_gamma = option_gamma(
    current_spy,
    last_strike,
    remaining_T,
    RISK_FREE_RATE,
    current_vol,
)

current_delta_notional_eur = (
    last_contracts
    * OPTION_MULTIPLIER
    * current_spy
    * abs(current_put_delta)
    / current_fx
)

current_effective_hedge_ratio = (
    current_delta_notional_eur
    / (
        current_equity
        * current_beta
    )
)

stress_rows = []

for scenario in stress_scenarios:

    # ---------------------------------------------------------
    # Equity + FX shock
    # EURUSD is USD per EUR.
    # Positive EURUSD shock = stronger EUR = lower EUR value of USD assets.
    # ---------------------------------------------------------

    stressed_fx = (
        current_fx
        * (
            1
            + scenario["EURUSD Shock"]
        )
    )

    fx_factor = (
        current_fx
        / stressed_fx
    )

    stressed_equity = (
        current_equity
        * (
            1
            + scenario["Equity Shock"]
        )
        * fx_factor
    )

    equity_pnl = (
        stressed_equity
        - current_equity
    )

    # ---------------------------------------------------------
    # Bonds: duration + convexity
    # ---------------------------------------------------------

    stressed_bonds = 0.0

    for _, row in bonds.iterrows():

        issuer = row["Issuer"]

        current_bond = float(
            bond_detail.loc[
                current_date,
                issuer
            ]
        )

        dy_stress = float(
            scenario[
                "Rate Shock"
            ]
        )

        price_change = (
            -row["Modified Duration"]
            * dy_stress
            + 0.5
            * row["Convexity"]
            * dy_stress**2
        )

        stressed_bonds += (
            current_bond
            * (
                1
                + price_change
            )
        )

    bond_pnl = (
        stressed_bonds
        - current_bonds
    )

    # ---------------------------------------------------------
    # Option: Black-Scholes repricing
    # ---------------------------------------------------------

    stressed_spy = (
        current_spy
        * (
            1
            + scenario["Equity Shock"]
        )
    )

    stressed_vol = float(
        np.clip(
            current_vol
            * scenario[
                "Vol Multiplier"
            ],
            MIN_VOL,
            MAX_VOL
        )
    )

    stressed_put_usd = bs_put(
        stressed_spy,
        last_strike,
        remaining_T,
        RISK_FREE_RATE,
        stressed_vol,
    )

    stressed_option_eur = (
        stressed_put_usd
        * OPTION_MULTIPLIER
        * last_contracts
        / stressed_fx
    )

    option_pnl = (
        stressed_option_eur
        - current_options
    )

    stressed_delta = put_delta(
        stressed_spy,
        last_strike,
        remaining_T,
        RISK_FREE_RATE,
        stressed_vol,
    )

    stressed_gamma = option_gamma(
        stressed_spy,
        last_strike,
        remaining_T,
        RISK_FREE_RATE,
        stressed_vol,
    )

    stressed_delta_notional_eur = (
        last_contracts
        * OPTION_MULTIPLIER
        * stressed_spy
        * abs(stressed_delta)
        / stressed_fx
    )

    stressed_hedge_ratio = (
        stressed_delta_notional_eur
        / (
            stressed_equity
            * current_beta
        )
        if stressed_equity > 0
        else np.nan
    )

    # ---------------------------------------------------------
    # CDS repricing
    # ---------------------------------------------------------

    stressed_cds_spread = (
        current_cds_spread
        + scenario[
            "Credit Spread Shock bps"
        ]
    )

    stressed_cds_raw = cds_value(
        CDS_NOTIONAL,
        CDS_CONTRACT_COUPON_BPS,
        stressed_cds_spread,
        CDS_MATURITY,
        RISK_FREE_RATE,
        CDS_RECOVERY,
    )

    stressed_cds_mtm = (
        stressed_cds_raw
        - cds_initial_residual
    )

    cds_pnl = (
        stressed_cds_mtm
        - current_cds_mtm
    )

    # ---------------------------------------------------------
    # P&L
    # ---------------------------------------------------------

    unhedged_pnl = (
        equity_pnl
        + bond_pnl
    )

    hedged_pnl = (
        unhedged_pnl
        + option_pnl
        + cds_pnl
    )

    hedge_benefit = (
        hedged_pnl
        - unhedged_pnl
    )

    hedge_effectiveness = (
        hedge_benefit
        / abs(unhedged_pnl)
        if unhedged_pnl != 0
        else np.nan
    )

    stressed_nav = (
        current_nav
        + hedged_pnl
    )

    stress_rows.append({
        "Scenario": scenario["Scenario"],
        "Equity P&L EUR": equity_pnl,
        "Bond P&L EUR": bond_pnl,
        "Option P&L EUR": option_pnl,
        "CDS P&L EUR": cds_pnl,
        "Unhedged P&L EUR": unhedged_pnl,
        "Hedged P&L EUR": hedged_pnl,
        "Hedge Benefit EUR": hedge_benefit,
        "Hedge Effectiveness": hedge_effectiveness,
        "Stressed Put Delta": stressed_delta,
        "Stressed Put Gamma": stressed_gamma,
        "Stressed Hedge Ratio": stressed_hedge_ratio,
        "Stressed NAV EUR": stressed_nav,
    })

multi_asset_stress = pd.DataFrame(
    stress_rows
)

# =============================================================================
# 17. RISK TRIGGERS
# =============================================================================

current_equity_weights = (
    equity_values_eur
    .loc[current_date]
    / current_equity
)

triggers = []

if (
    max_drawdown
    < RISK_LIMITS[
        "max_drawdown"
    ]
):
    triggers.append(
        "MAX DRAWDOWN LIMIT BREACHED"
    )

if (
    var99
    / current_nav
    > RISK_LIMITS[
        "var_pct"
    ]
):
    triggers.append(
        "TOTAL PORTFOLIO VAR LIMIT BREACHED"
    )

if (
    current_equity_weights.max()
    > RISK_LIMITS[
        "single_stock_weight"
    ]
):
    triggers.append(
        "EQUITY CONCENTRATION LIMIT BREACHED"
    )

if (
    current_general_cash
    / current_nav
    < RISK_LIMITS[
        "minimum_general_cash_pct"
    ]
):
    triggers.append(
        "MINIMUM GENERAL CASH BUFFER BREACHED"
    )

if (
    current_derivative_reserve
    < 0
):
    triggers.append(
        "DERIVATIVE RESERVE EXHAUSTED"
    )

# =============================================================================
# 18. MASTER DASHBOARD
# =============================================================================

dashboard = pd.DataFrame({
    "Metric": [
        "Target Initial Capital",
        "Actual Initial NAV",
        "Current Total NAV",
        "Current Equity NAV",
        "Current Bond NAV",
        "Current Option MTM",
        "Current CDS MTM",
        "Current General Cash",
        "Current Derivative Reserve",
        "Total Liquid Cash",
        "Total Dividends Received",
        "Total Bond Coupons Received",
        "Total CDS Premiums Paid",
        "Annualized Total Return",
        "Annualized Total Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "99% Historical VaR",
        "99% Historical ES",
        "Maximum Drawdown",
        "Equity Tracking Error",
        "Equity Information Ratio",
        "Equity Beta vs S&P 500",
        "Equity CAPM Alpha",
        "Total Tracking Error vs Blended Benchmark",
        "Total Information Ratio vs Blended Benchmark",
        "Current SPY Put Contracts",
        "Current Put Delta",
        "Current Put Gamma",
        "Current Effective Hedge Ratio",
        "Target Hedge Ratio",
        "Current CDS Spread",
        "CDS Contract Coupon",
        "CDS Data Source",
        "Current CDS Exposure",
        "CVA",
        "Credit-Adjusted CDS",
    ],
    "Value": [
        f"EUR {INITIAL_CAPITAL:,.2f}",
        f"EUR {initial_nav_actual:,.2f}",
        f"EUR {current_nav:,.2f}",
        f"EUR {current_equity:,.2f}",
        f"EUR {current_bonds:,.2f}",
        f"EUR {current_options:,.2f}",
        f"EUR {current_cds_mtm:,.2f}",
        f"EUR {current_general_cash:,.2f}",
        f"EUR {current_derivative_reserve:,.2f}",
        f"EUR {(current_general_cash + current_derivative_reserve):,.2f}",
        f"EUR {dividend_cash.sum():,.2f}",
        f"EUR {bond_coupon_cash.sum():,.2f}",
        f"EUR {-cds_premium_cash.sum():,.2f}",
        f"{ann_return:.2%}",
        f"{ann_vol:.2%}",
        f"{sharpe:.2f}",
        f"{sortino:.2f}",
        f"EUR {var99:,.2f}",
        f"EUR {es99:,.2f}",
        f"{max_drawdown:.2%}",
        f"{equity_tracking_error:.2%}",
        f"{equity_information_ratio:.2f}",
        f"{equity_beta:.2f}",
        f"{equity_capm_alpha:.2%}",
        f"{total_tracking_error:.2%}",
        f"{total_information_ratio:.2f}",
        f"{last_contracts}",
        f"{current_put_delta:.4f}",
        f"{current_put_gamma:.6f}",
        f"{current_effective_hedge_ratio:.2%}",
        f"{TARGET_HEDGE_RATIO:.2%}",
        f"{current_cds_spread:.1f} bps",
        f"{CDS_CONTRACT_COUPON_BPS:.1f} bps",
        cds_source,
        f"EUR {current_exposure:,.2f}",
        f"EUR {cva:,.2f}",
        f"EUR {credit_adjusted_cds:,.2f}",
    ]
})

print("\n" + "=" * 92)
print("MASTER PORTFOLIO DASHBOARD")
print("=" * 92)
print(
    dashboard.to_string(
        index=False
    )
)

print("\n" + "=" * 92)
print("CURRENT EQUITY WEIGHTS")
print("=" * 92)
print(
    (
        current_equity_weights
        * 100
    )
    .sort_values(
        ascending=False
    )
)

print("\n" + "=" * 92)
print("CDS-ONLY STRESS TEST")
print("=" * 92)
print(
    cds_only_stress
)

print("\n" + "=" * 92)
print("FULL MULTI-ASSET STRESS TEST")
print("=" * 92)
print(
    multi_asset_stress
)

print("\n" + "=" * 92)
print("ACTIVE RISK TRIGGERS")
print("=" * 92)

if triggers:
    for trigger in triggers:
        print(
            "-",
            trigger
        )
else:
    print(
        "No active risk triggers."
    )

dashboard.to_csv(
    OUTPUT_DIR / "dashboard.csv",
    index=False
)

cds_only_stress.to_csv(
    OUTPUT_DIR / "cds_only_stress.csv",
    index=False
)

multi_asset_stress.to_csv(
    OUTPUT_DIR / "multi_asset_stress.csv",
    index=False
)

# =============================================================================
# 19. CHARTS
# =============================================================================

plt.figure(figsize=(12, 6))
plt.plot(
    total_nav.index,
    total_nav,
    label="Total Portfolio NAV"
)
plt.title(
    "Total Multi-Asset Portfolio NAV"
)
plt.xlabel("Date")
plt.ylabel("EUR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    portfolio_growth.index,
    portfolio_growth,
    label="Portfolio"
)
plt.plot(
    blended_growth.index,
    blended_growth,
    label="Blended Benchmark"
)
plt.title(
    "Total Portfolio vs Blended Benchmark"
)
plt.xlabel("Date")
plt.ylabel("Growth of 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    drawdown.index,
    drawdown
)
plt.title(
    "Total Portfolio Drawdown"
)
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(
    multi_asset_stress[
        "Scenario"
    ],
    multi_asset_stress[
        "Unhedged P&L EUR"
    ]
)
plt.xticks(
    rotation=30,
    ha="right"
)
plt.title(
    "Unhedged P&L by Stress Scenario"
)
plt.ylabel("EUR")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(
    multi_asset_stress[
        "Scenario"
    ],
    multi_asset_stress[
        "Hedged P&L EUR"
    ]
)
plt.xticks(
    rotation=30,
    ha="right"
)
plt.title(
    "Hedged P&L by Stress Scenario"
)
plt.ylabel("EUR")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    cds_spread.index,
    cds_spread
)
plt.title(
    f"CDS Spread Series ({cds_source})"
)
plt.xlabel("Date")
plt.ylabel("Basis Points")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    general_cash.index,
    general_cash,
    label="General Cash"
)
plt.plot(
    derivative_reserve.index,
    derivative_reserve,
    label="Derivative Reserve"
)
plt.title(
    "Liquidity Accounts Through Time"
)
plt.xlabel("Date")
plt.ylabel("EUR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(
    "\nOutputs saved to:",
    OUTPUT_DIR.resolve()
)
