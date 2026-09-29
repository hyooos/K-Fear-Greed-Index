"""Walk-forward symmetric EGARCH(1,1)-t, same settings as the paper (robust_egarch):
refit every day on returns through t-1, min 252 obs, fallback to 60-day std when the fit
fails, gives < 1e-4 or > 0.20. Aligned to the KOSPI200 calendar of the final dataset.
Usage: python egarch_asset.py <SYMBOL>"""
import sys
import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
sym = sys.argv[1]
H = Path(__file__).parent
raw = pd.read_csv(H.parent / "01_final_dataset" / "KFG_final_10y.csv", parse_dates=["date"])
px = pd.read_csv(H / f"data/naver_{sym}.csv", parse_dates=["date"])
d = raw[["date"]].merge(px[["date", "close"]], on="date", how="left")
lr = np.log(d["close"] / d["close"].shift(1)).fillna(0)
n = len(lr)
sig = np.full(n, np.nan); fb = np.zeros(n, bool)
for t in range(252, n):
    tr = lr.iloc[:t]
    fallback = tr.iloc[-60:].std()
    val = np.nan
    try:
        res = arch_model(tr * 100, vol="EGARCH", p=1, q=1, dist="t", rescale=True).fit(disp="off", show_warning=False, options={"maxiter": 200})
        val = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])) / 100
    except Exception:
        pass
    if (not np.isfinite(val)) or val < 1e-4 or val > 0.20:
        val = fallback; fb[t] = True
    sig[t] = val
    if t % 250 == 0:
        print(sym, t, n, "fallback", fb[:t + 1].sum(), flush=True)
pd.DataFrame({"date": d["date"], "close": d["close"], "egarch_vol": sig, "fallback_used": fb}).to_csv(H / f"data/egarch_{sym}.csv", index=False)
print("done", sym, "fallback", fb.sum())
