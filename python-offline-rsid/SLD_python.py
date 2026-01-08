"""
Gangadhar Ambati

Python detector: "RSI Divergence — Early Structural (Non-Repainting)"
--------------------------------------------------------------------

What this script does
---------------------
This program implements TradingView’s built-in RSI Divergence logic (Regular divergences by default),
but exposes TWO timestamps for each divergence event:

1) TPYc_Date (candidate / earliest actionable date):
   - The divergence pivot bar date (pivot2 bar).
   - This is the earliest moment a trader could consider the structure *forming*.
   - In TradingView’s built-in indicator, this pivot is not *confirmed* until lbR future bars exist,
     which is why TV often appears "late" in replay.

2) TPY_Confirmed_Date (confirmed / TV-visible date):
   - The date of pivot confirmation = pivot2 + lbR bars.
   - This corresponds to when TV has enough future bars to confirm and draw the divergence line.

IMPORTANT GUARANTEE
-------------------
Signals are generated ONLY from the 'Prices' sheet (OHLC + Date).
Manual/TV/TLI columns (your analyst benchmark) are used ONLY for matching/evaluation downstream.

Inputs expected
---------------
An Excel workbook with a sheet named 'Prices' containing at least:
  Date, Open, High, Low, Close

Outputs
-------
A DataFrame of divergence events, anchored to pivot2 (event-level divergences), including:
  - Side: Bullish/Bearish
  - Type: Regular/Hidden
  - Pivot1_Date / Pivot2_Date
  - TPYc_Date / TPY_Confirmed_Date
  - RSI and Price values at pivots
  - Confirm_Gap_minus1 (TV’s rangeLower/rangeUpper constraint equivalent)

Author: Gangadhar Ambati
Generated: 2025-12-31
"""

import pandas as pd
import numpy as np


def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute Wilder RSI using RMA-style smoothing (equivalent to Pine `ta.rsi`).

    Parameters
    ----------
    close : np.ndarray
        Close price series.
    period : int
        RSI period (default 14).

    Returns
    -------
    np.ndarray
        RSI series aligned to `close` length.
        The first `period` values are NaN by design (warm-up / insufficient history).

    Notes (why this matches TradingView conceptually)
    -----------------------------------------------
    TradingView’s `ta.rsi` uses Wilder’s smoothing (RMA).
    We seed the initial average gain/loss using an SMA over the first `period` deltas,
    then apply Wilder’s recursive update for each subsequent bar.
    """
    close = np.asarray(close, dtype=float)

    # Day-to-day changes (delta). `prepend=close[0]` keeps array length unchanged.
    d = np.diff(close, prepend=close[0])

    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)

    rsi = np.full_like(close, np.nan, dtype=float)

    # Not enough data to compute RSI.
    if len(close) <= period:
        return rsi

    # Seed with simple average over the first `period` bars (after the first delta).
    avg_gain = gains[1 : period + 1].mean()
    avg_loss = losses[1 : period + 1].mean()

    # First RSI computed at index = period.
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi[period] = 100 - 100 / (1 + rs)

    # Wilder smoothing for the rest.
    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi[i] = 100 - 100 / (1 + rs)

    return rsi


def pivots(series: np.ndarray, lbL: int, lbR: int, mode: str) -> list[tuple[int, int]]:
    """
    Identify RSI pivots in the same spirit as TradingView:
      - `ta.pivotlow(series, lbL, lbR)`
      - `ta.pivothigh(series, lbL, lbR)`

    A pivot is defined at pivot index p, but it becomes *confirmed/known* only at:
        confirm index c = p + lbR

    Parameters
    ----------
    series : np.ndarray
        The oscillator series (here: RSI).
    lbL, lbR : int
        Left and right lookback window sizes.
    mode : str
        "low"  -> pivot low
        "high" -> pivot high

    Returns
    -------
    list[tuple[int, int]]
        List of (pivot_index, confirm_index).

    Practical meaning
    -----------------
    - pivot_index (p) is the candle where pivot happens (structural anchor).
    - confirm_index (c) is when you *can be sure* it was a pivot (needs lbR future bars).
    """
    out: list[tuple[int, int]] = []
    n = len(series)

    # Pivot index p must have lbL bars to the left and lbR to the right.
    for p in range(lbL, n - lbR):
        v = series[p]
        if np.isnan(v):
            continue

        left = series[p - lbL : p]
        right = series[p + 1 : p + lbR + 1]

        if mode == "low":
            # pivot low: v <= all neighbors in the window
            if np.all(v <= left) and np.all(v <= right):
                out.append((p, p + lbR))

        elif mode == "high":
            # pivot high: v >= all neighbors in the window
            if np.all(v >= left) and np.all(v >= right):
                out.append((p, p + lbR))

        else:
            raise ValueError("mode must be 'low' or 'high'")

    return out


def tv_rsi_divergences(
    prices: pd.DataFrame,
    rsi_len: int = 14,
    lbL: int = 5,
    lbR: int = 5,
    rangeLower: int = 5,
    rangeUpper: int = 60,
    plotHidden: bool = False,
) -> pd.DataFrame:
    """
    Detect RSI divergences using TradingView built-in logic semantics.

    This replicates the core Pine structure:

        plFound = not na(ta.pivotlow(osc, lbL, lbR))
        phFound = not na(ta.pivothigh(osc, lbL, lbR))

        oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and inRange(plFound[1])
        priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)

        bullCond = priceLL and oscHL and plFound

    In English:
    -----------
    - Only evaluate divergence when an RSI pivot is confirmed (needs lbR future bars).
    - Compare the current RSI pivot to the previous RSI pivot of the same type.
    - Enforce a pivot-to-pivot spacing constraint (rangeLower..rangeUpper) measured in bars.
    - Regular divergences:
        Bullish Regular: price lower low, RSI higher low (pivot lows)
        Bearish Regular: price higher high, RSI lower high (pivot highs)
    - Hidden divergences (optional):
        Bullish Hidden: price higher low, RSI lower low
        Bearish Hidden: price lower high, RSI higher high

    Output timestamps
    -----------------
    - Pivot2_Date: the pivot bar (event anchor)
    - TPYc_Date: same as Pivot2_Date (earliest “structural” date)
    - TPY_Confirmed_Date: pivot confirmation date (Pivot2_Date + lbR bars)

    Returns
    -------
    pd.DataFrame
        One row per divergence event.
    """
    df = prices.copy()

    # Defensive: parse/normalize dates and sort chronologically.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    dates = df["Date"]

    # RSI oscillator (Wilder)
    osc = rsi_wilder(close, rsi_len)

    # RSI pivots, sorted by confirm index so "previous pivot" is well-defined.
    pl = sorted(pivots(osc, lbL, lbR, "low"), key=lambda t: t[1])
    ph = sorted(pivots(osc, lbL, lbR, "high"), key=lambda t: t[1])

    events: list[dict] = []

    def add_event(side: str, dtype: str, p1: int, c1: int, p2: int, c2: int) -> None:
        """
        Append a divergence event to output list.

        side = "Bullish" or "Bearish"
        dtype = "Regular" or "Hidden"

        p1/p2 are pivot bar indices (structure anchors)
        c1/c2 are confirmation indices (when pivot becomes known)
        """
        events.append(
            {
                "Side": side,
                "Type": dtype,
                "Pivot1_Index": int(p1),
                "Pivot2_Index": int(p2),
                "Pivot1_Date": dates.iloc[p1],
                "Pivot2_Date": dates.iloc[p2],
                # Candidate/early = pivot bar itself (structural anchor)
                "TPYc_Date": dates.iloc[p2],
                # Confirmed/late = pivot becomes known only after lbR future bars
                "TPY_Confirmed_Date": dates.iloc[c2],
                "RSI_P1": float(osc[p1]),
                "RSI_P2": float(osc[p2]),
                # Use low for bullish (pivot lows), high for bearish (pivot highs)
                "Price_P1": float(low[p1] if side == "Bullish" else high[p1]),
                "Price_P2": float(low[p2] if side == "Bullish" else high[p2]),
                # Pine-equivalent spacing metric: barssince(previousPivotFound) minus 1
                "Confirm_Gap_minus1": int(c2 - c1 - 1),
            }
        )

    # --------------------- Bullish (pivot lows) ---------------------
    for i in range(1, len(pl)):
        p1, c1 = pl[i - 1]
        p2, c2 = pl[i]

        # Pivot spacing constraint (replicates _inRange(plFound[1]))
        gap = c2 - c1 - 1
        if not (rangeLower <= gap <= rangeUpper):
            continue

        # Regular Bullish: price LL, RSI HL
        if (low[p2] < low[p1]) and (osc[p2] > osc[p1]):
            add_event("Bullish", "Regular", p1, c1, p2, c2)

        # Hidden Bullish (optional): price HL, RSI LL
        if plotHidden and (low[p2] > low[p1]) and (osc[p2] < osc[p1]):
            add_event("Bullish", "Hidden", p1, c1, p2, c2)

    # --------------------- Bearish (pivot highs) ---------------------
    for i in range(1, len(ph)):
        p1, c1 = ph[i - 1]
        p2, c2 = ph[i]

        gap = c2 - c1 - 1
        if not (rangeLower <= gap <= rangeUpper):
            continue

        # Regular Bearish: price HH, RSI LH
        if (high[p2] > high[p1]) and (osc[p2] < osc[p1]):
            add_event("Bearish", "Regular", p1, c1, p2, c2)

        # Hidden Bearish (optional): price LH, RSI HH
        if plotHidden and (high[p2] < high[p1]) and (osc[p2] > osc[p1]):
            add_event("Bearish", "Hidden", p1, c1, p2, c2)

    out = (
        pd.DataFrame(events)
        .sort_values(["Side", "Type", "Pivot2_Date"])
        .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    # --------------------- Example: run against your workbook ---------------------
    # NOTE: This block is safe to delete in production; it is only a convenience runner.
    xlsx = "RSID_GA_NIFTY_Daily_Observations.xlsx"
    prices = pd.read_excel(xlsx, sheet_name="Prices")
    events = tv_rsi_divergences(prices, plotHidden=False)

    print("Detected divergences (Regular):", len(events))
    events.to_csv("RSI_Early_Divergence_Signals.csv", index=False)
