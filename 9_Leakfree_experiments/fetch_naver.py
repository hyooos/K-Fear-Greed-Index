"""Download daily index closes from Naver finance chart API (no login)."""
import re, subprocess, sys
import pandas as pd
from pathlib import Path
OUT = Path(__file__).parent / "data"
for sym in sys.argv[1:]:
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol={sym}&timeframe=day&count=4000&requestType=0"
    raw = subprocess.run(["curl", "-s", "-m", "30", url], capture_output=True).stdout.decode("euc-kr", "ignore")
    rows = [x.split("|") for x in re.findall(r'item data="([^"]+)"', raw)]
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    df.to_csv(OUT / f"naver_{sym}.csv", index=False)
    print(sym, len(df), df.date.min().date(), df.date.max().date())
