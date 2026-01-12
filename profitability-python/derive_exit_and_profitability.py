"""
derive_exit_and_profitability_sagar_logic.py

Implements rule-based exit logic for divergence trades based on
Dr. Sagar Bansal's RSI exhaustion and breakeven principles, and
computes realized profitability for Manual, SLD, TradingView, and TLI signals.

Used for Section 4.5 and 4.6 of the PhD thesis.
"""
import pandas as pd
import numpy as np

INFILE = "RSID_GA_NIFTY_Daily_Observations.xlsx"
OUTFILE = "RSID_GA_NIFTY_Daily_Observations_exit_v5_sagarlogic_v52.xlsx"

OBS_SHEET = "NIFTY_Daily"
PRICES_SHEET = "Prices"

RSI_PERIOD = 14
MAX_HOLD_BARS = 120
BREAKEVEN_FAVORABLE_PCT = 1.0
MIN_BE_BARS = 5

REASON_BUY_RSI = "RSI70_Down"
REASON_SELL_RSI = "RSI30_Up"
REASON_BE = "BreakevenStop"
REASON_MH = f"MaxHold({MAX_HOLD_BARS})"


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    rsi = pd.Series(np.nan, index=close.index, dtype=float)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    ag = al = np.nan
    for i in range(len(close)):
        if i < period:
            continue
        if i == period:
            ag = avg_gain.iat[i]
            al = avg_loss.iat[i]
        else:
            ag = (ag * (period - 1) + gain.iat[i]) / period
            al = (al * (period - 1) + loss.iat[i]) / period

        if al == 0:
            rsi.iat[i] = 100.0
        else:
            rs = ag / al
            rsi.iat[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def next_trading_index(prices: pd.DataFrame, dt: pd.Timestamp):
    if pd.isna(dt):
        return None
    pos = prices["Date"].searchsorted(dt, side="right")
    return int(pos) if pos < len(prices) else None


def pct_profit(trade_type: str, entry_open: float, exit_close: float) -> float:
    if pd.isna(entry_open) or pd.isna(exit_close) or entry_open == 0:
        return np.nan
    if trade_type == "Buy":
        return (exit_close - entry_open) / entry_open * 100.0
    if trade_type == "Sell":
        return (entry_open - exit_close) / entry_open * 100.0
    return np.nan


def find_rsi_exit(prices, rsi, entry_idx, end_idx, trade_type):
    """Pass-1: if RSI exhaustion happens anywhere in window, return earliest exhaustion exit index."""
    prev = float(rsi.iat[entry_idx - 1]) if entry_idx > 0 and not pd.isna(rsi.iat[entry_idx - 1]) else np.nan

    if trade_type == "Buy":
        crossed_up = False
        for i in range(entry_idx, end_idx + 1):
            cur = float(rsi.iat[i]) if not pd.isna(rsi.iat[i]) else np.nan
            if not pd.isna(prev) and not pd.isna(cur):
                if (not crossed_up) and (prev <= 70) and (cur > 70):
                    crossed_up = True
                if crossed_up and (prev >= 70) and (cur < 70):
                    return i
            prev = cur

    elif trade_type == "Sell":
        crossed_down = False
        for i in range(entry_idx, end_idx + 1):
            cur = float(rsi.iat[i]) if not pd.isna(rsi.iat[i]) else np.nan
            if not pd.isna(prev) and not pd.isna(cur):
                if (not crossed_down) and (prev >= 30) and (cur < 30):
                    crossed_down = True
                if crossed_down and (prev <= 30) and (cur > 30):
                    return i
            prev = cur

    return None


def find_breakeven_exit(prices, entry_idx, end_idx, trade_type, entry_open):
    """
    Pass-2: BE only if no RSI exit was found.
    Logic: once trade becomes favorable by >= 1%,
           after MIN_BE_BARS, exit when Close returns to entry (<= entry for Buy; >= entry for Sell)
    """
    armed = False

    for i in range(entry_idx, end_idx + 1):
        close_i = float(prices.at[i, "Close"])

        # arm after favorable move
        if not armed:
            if trade_type == "Buy" and close_i >= entry_open * (1 + BREAKEVEN_FAVORABLE_PCT / 100.0):
                armed = True
            if trade_type == "Sell" and close_i <= entry_open * (1 - BREAKEVEN_FAVORABLE_PCT / 100.0):
                armed = True

        if armed and (i - entry_idx) >= MIN_BE_BARS:
            if trade_type == "Buy" and close_i <= entry_open:
                return i
            if trade_type == "Sell" and close_i >= entry_open:
                return i

    return None


def derive_exit_sagar_v52(prices, rsi, entry_idx, trade_type):
    n = len(prices)
    entry_open = float(prices.at[entry_idx, "Open"])
    end_idx = min(entry_idx + MAX_HOLD_BARS, n - 1)

    # PASS 1: RSI exhaustion has priority
    rsi_exit_idx = find_rsi_exit(prices, rsi, entry_idx, end_idx, trade_type)
    if rsi_exit_idx is not None:
        return prices.at[rsi_exit_idx, "Date"], float(prices.at[rsi_exit_idx, "Close"]), (
            REASON_BUY_RSI if trade_type == "Buy" else REASON_SELL_RSI
        )

    # PASS 2: BreakevenStop if no RSI exit
    be_exit_idx = find_breakeven_exit(prices, entry_idx, end_idx, trade_type, entry_open)
    if be_exit_idx is not None:
        return prices.at[be_exit_idx, "Date"], float(prices.at[be_exit_idx, "Close"]), REASON_BE

    # PASS 3: MaxHold fallback
    return prices.at[end_idx, "Date"], float(prices.at[end_idx, "Close"]), REASON_MH


def main():
    obs = pd.read_excel(INFILE, sheet_name=OBS_SHEET)
    prices = pd.read_excel(INFILE, sheet_name=PRICES_SHEET)

    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close"]:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    rsi = rsi_wilder(prices["Close"], RSI_PERIOD)

    obs["Confirmation Date"] = pd.to_datetime(obs["Confirmation Date"], errors="coerce")
    obs["Trade Type"] = obs["Trade Type"].astype(str).str.strip()

    # parse optional signal date cols if present
    for c in ["SLD_Date", "T1_TV_Date ", "T1_TV_Date", "TLI_Date (TTLI)", "TLI_Date"]:
        if c in obs.columns:
            obs[c] = pd.to_datetime(obs[c], errors="coerce")

    # ensure output cols
    needed = [
        "Exit_Date_Analyst", "Exit_Price_Analyst", "Exit_Reason",
        "Entry_Date_Analyst", "Entry_Open_Analyst",
        "Entry_Date_SLD", "Entry_Open_SLD",
        "Entry_Date_TV", "Entry_Open_TV",
        "Entry_Date_TLI", "Entry_Open_TLI",
        "Profit_%", "Profit_SLD_%", "Profit_TV_%", "Profit_TLI_%",
        "Delta_SLD_vs_Manual_%", "Delta_SLD_vs_TV_%", "Delta_TLI_vs_Manual_%"
    ]
    for c in needed:
        if c not in obs.columns:
            obs[c] = np.nan

    def entry_from_signal(dt):
        idx = next_trading_index(prices, dt)
        if idx is None:
            return pd.NaT, np.nan, None
        return prices.at[idx, "Date"], float(prices.at[idx, "Open"]), idx

    # manual exit based on Confirmation Date (analyst benchmark)
    for i in range(len(obs)):
        sig = obs.at[i, "Confirmation Date"]
        tt = obs.at[i, "Trade Type"]
        if pd.isna(sig) or tt not in ("Buy", "Sell"):
            continue

        ed, eo, eidx = entry_from_signal(sig)
        obs.at[i, "Entry_Date_Analyst"] = ed
        obs.at[i, "Entry_Open_Analyst"] = eo
        if eidx is None:
            continue

        xd, xc, xr = derive_exit_sagar_v52(prices, rsi, eidx, tt)
        obs.at[i, "Exit_Date_Analyst"] = xd
        obs.at[i, "Exit_Price_Analyst"] = xc
        obs.at[i, "Exit_Reason"] = xr
        obs.at[i, "Profit_%"] = pct_profit(tt, eo, xc)

    # other methods: own entry date, common exit
    def fill(sig_col, out_date, out_open, out_profit):
        if sig_col not in obs.columns:
            return
        for i in range(len(obs)):
            sig = obs.at[i, sig_col]
            tt = obs.at[i, "Trade Type"]
            xc = obs.at[i, "Exit_Price_Analyst"]
            if pd.isna(sig) or tt not in ("Buy", "Sell") or pd.isna(xc):
                continue
            ed, eo, _ = entry_from_signal(sig)
            obs.at[i, out_date] = ed
            obs.at[i, out_open] = eo
            obs.at[i, out_profit] = pct_profit(tt, eo, float(xc))

    fill("SLD_Date", "Entry_Date_SLD", "Entry_Open_SLD", "Profit_SLD_%")

    tv_col = "T1_TV_Date " if "T1_TV_Date " in obs.columns else ("T1_TV_Date" if "T1_TV_Date" in obs.columns else None)
    if tv_col:
        fill(tv_col, "Entry_Date_TV", "Entry_Open_TV", "Profit_TV_%")

    tli_col = "TLI_Date (TTLI)" if "TLI_Date (TTLI)" in obs.columns else ("TLI_Date" if "TLI_Date" in obs.columns else None)
    if tli_col:
        fill(tli_col, "Entry_Date_TLI", "Entry_Open_TLI", "Profit_TLI_%")

    obs["Delta_SLD_vs_Manual_%"] = obs["Profit_SLD_%"] - obs["Profit_%"]
    obs["Delta_SLD_vs_TV_%"] = obs["Profit_SLD_%"] - obs["Profit_TV_%"]
    obs["Delta_TLI_vs_Manual_%"] = obs["Profit_TLI_%"] - obs["Profit_%"]

    with pd.ExcelWriter(OUTFILE, engine="openpyxl") as w:
        obs.to_excel(w, sheet_name=OBS_SHEET, index=False)
        prices.to_excel(w, sheet_name=PRICES_SHEET, index=False)

    print("Wrote:", OUTFILE)


if __name__ == "__main__":
    main()
