"""K-FGI sub-index calculator.

This script calculates the seven CNN-style Fear & Greed sub-indices used in
the project.  Every sub-index is converted to a 0-100 score where a higher
score means greed and a lower score means fear.

The direction rule follows the Hanwha report and the project notes:
- higher means greed: market momentum, stock price strength, stock breadth
- higher means fear: put/call ratio, volatility, safe-haven demand, junk spread

Project choices:
- sub_index3 uses rolling z-score + normal CDF by default because the oscillator
  can have large outliers.
- sub_index7 uses BBB- minus 3Y government bond yield by default to capture
  riskier credit demand.

Examples
--------
python 1_KFGI_subindex/calculate_subindices.py momentum \
  --input kospi200.csv --output sub_index1_momentum.csv

python 1_KFGI_subindex/calculate_subindices.py breadth \
  --ohlcv kospi_ohlcv.csv --output sub_index3_osc.csv

python 1_KFGI_subindex/calculate_subindices.py junk \
  --rates "시장금리(월,분기,년)_21204022.csv" --output sub_index7_junk.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252


DATE_CANDIDATES = ("date", "Date", "날짜", "Unnamed: 0", "일자")
CLOSE_CANDIDATES = ("close", "Close", "종가", "kospi_close", "KOSPI200", "KOSPI")
VOLUME_CANDIDATES = ("trading_value", "거래대금", "amount", "거래량", "volume", "Volume")


def read_csv_smart(path: str | Path) -> pd.DataFrame:
    """Read Korean/English CSV files without hard-coding one encoding."""
    path = Path(path)
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path)


def first_existing(columns: pd.Index, candidates: tuple[str, ...], name: str) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"{name} 컬럼을 찾지 못했습니다. 후보={candidates}, 현재={list(columns)}")


def normalize_date(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if date_col is None:
        date_col = first_existing(df.columns, DATE_CANDIDATES, "date")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def percentile_of_last(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    last = values[-1]
    return 100.0 * (values <= last).sum() / len(values)


def to_greed_score(
    raw: pd.Series,
    *,
    higher_is: str,
    window: int = TRADING_DAYS,
    min_periods: int | None = None,
    method: str = "percentile",
    clip_z: float = 5.0,
) -> pd.Series:
    """Convert a raw indicator into a 0-100 greed score.

    Parameters
    ----------
    raw:
        Raw indicator series.
    higher_is:
        "greed" when a high raw value means greed, "fear" when a high raw value
        means fear.  Fear-direction indicators are multiplied by -1 before
        scoring.
    window:
        Rolling lookback.  Use 252 for daily data and 12 for monthly data.
    method:
        "percentile" implements the report's recent-1-year percentile rank.
        "z_cdf" reproduces the notebook-style rolling z-score then normal CDF.
    """
    if higher_is not in {"greed", "fear"}:
        raise ValueError("higher_is must be 'greed' or 'fear'")
    if method not in {"percentile", "z_cdf"}:
        raise ValueError("method must be 'percentile' or 'z_cdf'")

    x = pd.to_numeric(raw, errors="coerce")
    greed_aligned = x if higher_is == "greed" else -x
    min_periods = window if min_periods is None else min_periods

    if method == "percentile":
        return greed_aligned.rolling(window, min_periods=min_periods).apply(
            percentile_of_last,
            raw=True,
        )

    mean = greed_aligned.rolling(window, min_periods=min_periods).mean()
    std = greed_aligned.rolling(window, min_periods=min_periods).std(ddof=1)
    z = ((greed_aligned - mean) / std.replace(0, np.nan)).clip(-clip_z, clip_z)
    return z.map(lambda v: np.nan if pd.isna(v) else normal_cdf(float(v)) * 100.0)


def save_output(df: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved: {output} rows={len(df):,}")


def maybe_drop_warmup(df: pd.DataFrame, score_col: str, keep_warmup: bool) -> pd.DataFrame:
    if keep_warmup:
        return df
    return df.loc[df[score_col].notna()].reset_index(drop=True)


def calc_market_momentum(args: argparse.Namespace) -> None:
    df = normalize_date(read_csv_smart(args.input), args.date_col)
    close_col = args.close_col or first_existing(df.columns, CLOSE_CANDIDATES, "close")
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df["ma125"] = df["close"].rolling(args.ma_window, min_periods=args.ma_window).mean()
    df["momentum_raw"] = df["close"] / df["ma125"] - 1.0
    df["Momentum_Score"] = to_greed_score(
        df["momentum_raw"],
        higher_is="greed",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    df = maybe_drop_warmup(df, "Momentum_Score", args.keep_warmup)
    save_output(df[["date", "close", "ma125", "momentum_raw", "Momentum_Score"]], args.output)


def calc_stock_strength(args: argparse.Namespace) -> None:
    df = normalize_date(read_csv_smart(args.ohlcv), args.date_col)
    ticker_col = args.ticker_col
    close_col = args.close_col or first_existing(df.columns, CLOSE_CANDIDATES, "close")
    if ticker_col not in df.columns:
        raise ValueError(f"ticker 컬럼이 없습니다: {ticker_col}")

    df[ticker_col] = df[ticker_col].astype(str).str.zfill(6)
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.sort_values([ticker_col, "date"]).reset_index(drop=True)

    grouped = df.groupby(ticker_col, group_keys=False)["close"]
    roll_high = grouped.rolling(args.high_low_window, min_periods=args.min_obs).max().reset_index(level=0, drop=True)
    roll_low = grouped.rolling(args.high_low_window, min_periods=args.min_obs).min().reset_index(level=0, drop=True)
    roll_count = grouped.rolling(args.high_low_window, min_periods=1).count().reset_index(level=0, drop=True)

    active = roll_count >= args.min_obs
    df["new_high"] = active & (df["close"] >= roll_high)
    df["new_low"] = active & (df["close"] <= roll_low)
    df["active"] = active

    daily = df.groupby("date").agg(
        new_high=("new_high", "sum"),
        new_low=("new_low", "sum"),
        active_count=("active", "sum"),
    ).reset_index()
    daily["strength_raw"] = (daily["new_high"] - daily["new_low"]) / daily["active_count"].replace(0, np.nan) * 100.0
    daily["Strength_Score"] = to_greed_score(
        daily["strength_raw"],
        higher_is="greed",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    daily = maybe_drop_warmup(daily, "Strength_Score", args.keep_warmup)
    save_output(daily, args.output)


def calc_breadth(args: argparse.Namespace) -> None:
    if args.oscillator:
        daily = normalize_date(read_csv_smart(args.oscillator), args.date_col)
        osc_col = args.osc_col or "oscillator"
        if osc_col not in daily.columns:
            raise ValueError(f"oscillator 컬럼이 없습니다: {osc_col}")
        daily["oscillator"] = pd.to_numeric(daily[osc_col], errors="coerce")
    else:
        df = normalize_date(read_csv_smart(args.ohlcv), args.date_col)
        ticker_col = args.ticker_col
        close_col = args.close_col or first_existing(df.columns, CLOSE_CANDIDATES, "close")
        value_col = args.value_col or first_existing(df.columns, VOLUME_CANDIDATES, "volume/value")
        if ticker_col not in df.columns:
            raise ValueError(f"ticker 컬럼이 없습니다: {ticker_col}")

        df[ticker_col] = df[ticker_col].astype(str).str.zfill(6)
        df["close"] = pd.to_numeric(df[close_col], errors="coerce")
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.sort_values([ticker_col, "date"]).reset_index(drop=True)
        df["prev_close"] = df.groupby(ticker_col)["close"].shift(1)

        valid = df["prev_close"].notna()
        up = valid & (df["close"] > df["prev_close"])
        down = valid & (df["close"] < df["prev_close"])
        df["up_value"] = df["value"].where(up, 0.0)
        df["down_value"] = df["value"].where(down, 0.0)

        daily = df.groupby("date", as_index=False)[["up_value", "down_value"]].sum()
        daily = daily.rename(columns={"up_value": "AV", "down_value": "DV"})
        daily["net_value"] = daily["AV"] - daily["DV"]
        daily["trend_fast"] = daily["net_value"].ewm(alpha=args.alpha_fast, adjust=False).mean()
        daily["trend_slow"] = daily["net_value"].ewm(alpha=args.alpha_slow, adjust=False).mean()
        daily["oscillator"] = daily["trend_fast"] - daily["trend_slow"]

    daily["osc_score_0_100"] = to_greed_score(
        daily["oscillator"],
        higher_is="greed",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    daily = maybe_drop_warmup(daily, "osc_score_0_100", args.keep_warmup)
    save_output(daily, args.output)


def calc_put_call(args: argparse.Namespace) -> None:
    df = normalize_date(read_csv_smart(args.input), args.date_col)
    if args.ratio_col:
        df["put_call_ratio"] = pd.to_numeric(df[args.ratio_col], errors="coerce")
    else:
        df["put_call_ratio"] = (
            pd.to_numeric(df[args.put_col], errors="coerce")
            / pd.to_numeric(df[args.call_col], errors="coerce").replace(0, np.nan)
        )
    df["score_pcr"] = to_greed_score(
        df["put_call_ratio"],
        higher_is="fear",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    df = maybe_drop_warmup(df, "score_pcr", args.keep_warmup)
    save_output(df[["date", "put_call_ratio", "score_pcr"]], args.output)


def calc_volatility(args: argparse.Namespace) -> None:
    df = normalize_date(read_csv_smart(args.input), args.date_col)
    vol_col = args.vol_col
    df["volatility"] = pd.to_numeric(df[vol_col], errors="coerce")
    df["vol_ma"] = df["volatility"].rolling(args.ma_window, min_periods=args.ma_window).mean()
    df["vol_ratio"] = df["volatility"] / df["vol_ma"]
    df["Score_Fear"] = to_greed_score(
        df["vol_ratio"],
        higher_is="fear",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    df = maybe_drop_warmup(df, "Score_Fear", args.keep_warmup)
    save_output(df[["date", "volatility", "vol_ma", "vol_ratio", "Score_Fear"]], args.output)


def calc_safe_demand(args: argparse.Namespace) -> None:
    df = normalize_date(read_csv_smart(args.input), args.date_col)
    stock = pd.to_numeric(df[args.stock_col], errors="coerce")
    bond = pd.to_numeric(df[args.bond_col], errors="coerce")
    df["stock_ret_20d"] = stock / stock.shift(args.return_window) - 1.0
    df["bond_ret_20d"] = bond / bond.shift(args.return_window) - 1.0
    df["safe_demand_raw"] = df["bond_ret_20d"] - df["stock_ret_20d"]
    df["score_safe_demand"] = to_greed_score(
        df["safe_demand_raw"],
        higher_is="fear",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    df = maybe_drop_warmup(df, "score_safe_demand", args.keep_warmup)
    save_output(df[["date", "stock_ret_20d", "bond_ret_20d", "safe_demand_raw", "score_safe_demand"]], args.output)


def load_rate_table(path: str | Path) -> pd.DataFrame:
    raw = read_csv_smart(path)
    if "계정항목" not in raw.columns:
        return normalize_date(raw)

    date_cols = list(raw.columns[4:])
    wanted = {
        "국고채(3년)": "GOV3Y",
        "회사채(3년, BBB-)": "BBB",
        "회사채(3년, AA-)": "AA",
    }
    series = []
    for item, out_col in wanted.items():
        part = raw.loc[raw["계정항목"] == item, date_cols]
        if part.empty:
            continue
        s = part.iloc[0].T.rename(out_col)
        series.append(s)
    if not series:
        raise ValueError("시장금리 원자료에서 국고채/회사채 행을 찾지 못했습니다.")
    df = pd.concat(series, axis=1).reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m", errors="coerce")
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def calc_junk_demand(args: argparse.Namespace) -> None:
    df = load_rate_table(args.rates)
    if args.spread_col:
        df["junk_spread"] = pd.to_numeric(df[args.spread_col], errors="coerce")
    else:
        if args.spread_kind == "AA_GOV":
            df["junk_spread"] = pd.to_numeric(df["AA"], errors="coerce") - pd.to_numeric(df["GOV3Y"], errors="coerce")
        elif args.spread_kind == "BBB_GOV":
            df["junk_spread"] = pd.to_numeric(df["BBB"], errors="coerce") - pd.to_numeric(df["GOV3Y"], errors="coerce")
        elif args.spread_kind == "BBB_AA":
            df["junk_spread"] = pd.to_numeric(df["BBB"], errors="coerce") - pd.to_numeric(df["AA"], errors="coerce")
        else:
            raise ValueError(f"unknown spread_kind: {args.spread_kind}")

    df["Final_JunkIndex"] = to_greed_score(
        df["junk_spread"],
        higher_is="fear",
        window=args.score_window,
        min_periods=args.min_periods,
        method=args.score_method,
    )
    df = maybe_drop_warmup(df, "Final_JunkIndex", args.keep_warmup)
    cols = ["date"] + [c for c in ("GOV3Y", "BBB", "AA") if c in df.columns] + ["junk_spread", "Final_JunkIndex"]
    save_output(df[cols], args.output)


def add_score_args(
    parser: argparse.ArgumentParser,
    *,
    default_window: int = TRADING_DAYS,
    default_method: str = "percentile",
) -> None:
    parser.add_argument("--score-window", type=int, default=default_window)
    parser.add_argument("--min-periods", type=int, default=None)
    parser.add_argument("--score-method", choices=["percentile", "z_cdf"], default=default_method)
    parser.add_argument("--keep-warmup", action="store_true", help="Keep rows where the rolling score is still NaN.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate K-FGI sub-index CSV files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("momentum", help="sub_index1: market momentum")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sub_index1_momentum.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--close-col", default=None)
    p.add_argument("--ma-window", type=int, default=125)
    add_score_args(p)
    p.set_defaults(func=calc_market_momentum)

    p = sub.add_parser("strength", help="sub_index2: 52-week high-low strength")
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--output", default="sub_index2_strength.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--ticker-col", default="ticker")
    p.add_argument("--close-col", default=None)
    p.add_argument("--high-low-window", type=int, default=TRADING_DAYS)
    p.add_argument("--min-obs", type=int, default=TRADING_DAYS)
    add_score_args(p)
    p.set_defaults(func=calc_stock_strength)

    p = sub.add_parser("breadth", help="sub_index3: McClellan volume/value oscillator")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ohlcv")
    source.add_argument("--oscillator")
    p.add_argument("--output", default="sub_index3_osc.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--ticker-col", default="ticker")
    p.add_argument("--close-col", default=None)
    p.add_argument("--value-col", default=None)
    p.add_argument("--osc-col", default=None)
    p.add_argument("--alpha-fast", type=float, default=0.10)
    p.add_argument("--alpha-slow", type=float, default=0.05)
    add_score_args(p, default_method="z_cdf")
    p.set_defaults(func=calc_breadth)

    p = sub.add_parser("putcall", help="sub_index4: put/call ratio")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sub_index4_pcratio.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--ratio-col", default=None)
    p.add_argument("--put-col", default="put_volume")
    p.add_argument("--call-col", default="call_volume")
    add_score_args(p)
    p.set_defaults(func=calc_put_call)

    p = sub.add_parser("volatility", help="sub_index5: volatility relative to moving average")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sub_index5_volatility.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--vol-col", required=True)
    p.add_argument("--ma-window", type=int, default=50)
    add_score_args(p)
    p.set_defaults(func=calc_volatility)

    p = sub.add_parser("safe", help="sub_index6: safe-haven demand")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sub_index6_safedemand.csv")
    p.add_argument("--date-col", default=None)
    p.add_argument("--stock-col", required=True)
    p.add_argument("--bond-col", required=True)
    p.add_argument("--return-window", type=int, default=20)
    add_score_args(p)
    p.set_defaults(func=calc_safe_demand)

    p = sub.add_parser("junk", help="sub_index7: junk/corporate bond spread demand")
    p.add_argument("--rates", required=True)
    p.add_argument("--output", default="sub_index7_junk.csv")
    p.add_argument("--spread-col", default=None)
    p.add_argument("--spread-kind", choices=["AA_GOV", "BBB_GOV", "BBB_AA"], default="BBB_GOV")
    add_score_args(p, default_window=12)
    p.set_defaults(func=calc_junk_demand)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
