"""Convert KRX Data Marketplace index CSVs (지수 > 주가지수 > 개별지수 시세 추이) into
data/naver_<NAME>.csv (date, close) so that egarch_asset.py / step6 / step7 can use them.

Put either one file per index   : data/krx/<NAME>.csv
or several files for one index  : data/krx/<NAME>/*.csv   (e.g. split by year)
Usage: python import_krx.py
"""
import pandas as pd
from pathlib import Path

H = Path(__file__).parent
SRC = H / "data" / "krx"


def read_one(f):
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            x = pd.read_csv(f, encoding=enc)
            break
        except Exception:
            continue
    dc = [c for c in x.columns if "일자" in c or c.lower() == "date"][0]
    cc = [c for c in x.columns if "종가" in c or c.lower() == "close"][0]
    return pd.DataFrame({"date": pd.to_datetime(x[dc].astype(str).str.replace("/", "-")),
                         "close": pd.to_numeric(x[cc].astype(str).str.replace(",", ""))})


items = {f.stem: [f] for f in SRC.glob("*.csv")}
items |= {p.name: sorted(p.glob("*.csv")) for p in SRC.iterdir() if p.is_dir()}
for name, files in sorted(items.items()):
    out = pd.concat([read_one(f) for f in files]).drop_duplicates("date").sort_values("date")
    out.to_csv(H / "data" / f"naver_{name}.csv", index=False)
    print(name, len(out), out["date"].min().date(), out["date"].max().date())
