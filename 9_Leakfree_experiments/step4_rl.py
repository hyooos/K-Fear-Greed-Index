"""Step 4: reinforcement learning (PPO, implemented in torch).

State  : standardized features (train stats) + previous weight
Action : weight in [0, 1] via sigmoid of a Gaussian
Reward : 100 * pnl - kappa * (100 * min(pnl, 0))^2, pnl = a_s * r[s+2] - fee * |a_s - a_{s-1}|
Walk-forward: retrain at the start of each calendar year on rows s <= first_row(Y) - 6.
Seeds are run separately; the ensemble averages the deterministic weights.
Usage: python step4_rl.py <kappa> <seed> [mode]
mode "free": weight = sigmoid(u); mode "overlay": weight = clip(base * 2 * sigmoid(u), 0, 1),
where base is the trend x vol-target x safety rule (no K-FGI).
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lab import *

torch.set_num_threads(2)
KAPPA = float(sys.argv[1])
SEED = int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "free"

d = load()
d["lrv1"] = np.log(d["r"].abs() + 1e-4)
d["legv"] = np.log(d["egv"] + 1e-4)
FEATS = FALL + ["lrv1", "legv"]
BASEW = rule_weights(d, np.full(len(d), 50.0), True, use_kfgi=False, use_regime=False).values.astype(np.float32)
d["basew"] = BASEW
if MODE == "overlay":
    FEATS = FEATS + ["basew"]
X = d[FEATS].values.astype(np.float32)
R2 = d["r"].shift(-2).values.astype(np.float32)  # return earned by decision at row s
valid = ~np.isnan(X).any(axis=1)
first_valid = int(np.argmax(valid))
years = list(range(2016, 2026))
EP_LEN, N_EP, UPDATES, EPOCHS, MB = 126, 16, 150, 4, 256
GAMMA, LAM, CLIP, ENT, LR = 0.9, 0.9, 0.2, 0.01, 3e-4


class Net(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(k + 1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.v = nn.Sequential(nn.Linear(k + 1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.logstd = nn.Parameter(torch.tensor([-0.5]))

    def dist(self, s):
        mu = self.pi(s).squeeze(-1)
        return torch.distributions.Normal(mu, self.logstd.exp().expand_as(mu))


def train_year(tr_idx, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -5, 5).astype(np.float32)
    net = Net(Z.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lo, hi = tr_idx[0], tr_idx[-1]
    for _ in range(UPDATES):
        starts = rng.integers(lo, max(lo + 1, hi - EP_LEN), size=N_EP)
        S, U, LP, RW, V = [], [], [], [], []
        prev = torch.zeros(N_EP)
        with torch.no_grad():
            for t in range(EP_LEN):
                rows = np.minimum(starts + t, hi)
                s = torch.cat([torch.from_numpy(Z[rows]), prev[:, None]], 1)
                dist = net.dist(s)
                u = dist.sample()
                a = torch.sigmoid(u)
                if MODE == "overlay":
                    a = torch.clamp(torch.from_numpy(np.nan_to_num(BASEW[rows])) * 2 * a, 0, 1)
                ret = torch.from_numpy(np.nan_to_num(R2[rows]))
                pnl = a * ret - FEE * (a - prev).abs()
                rew = 100 * pnl - KAPPA * (100 * torch.clamp(pnl, max=0)) ** 2
                S.append(s); U.append(u); LP.append(dist.log_prob(u)); RW.append(rew)
                V.append(net.v(s).squeeze(-1))
                prev = a
            V.append(torch.zeros(N_EP))
            adv = torch.zeros(EP_LEN, N_EP)
            g = torch.zeros(N_EP)
            for t in reversed(range(EP_LEN)):
                delta = RW[t] + GAMMA * V[t + 1] - V[t]
                g = delta + GAMMA * LAM * g
                adv[t] = g
            retn = adv + torch.stack(V[:-1])
        S = torch.cat(S); U = torch.cat(U); LP = torch.cat(LP)
        A = adv.reshape(-1); RT = retn.reshape(-1)
        A = (A - A.mean()) / (A.std() + 1e-8)
        n = len(S)
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, MB):
                j = perm[i:i + MB]
                dist = net.dist(S[j])
                ratio = (dist.log_prob(U[j]) - LP[j]).exp()
                l1 = ratio * A[j]
                l2 = ratio.clamp(1 - CLIP, 1 + CLIP) * A[j]
                loss = -torch.min(l1, l2).mean() + 0.5 * (net.v(S[j]).squeeze(-1) - RT[j]).pow(2).mean() \
                    - ENT * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    return net, Z


w = np.full(len(d), np.nan)
prev = 0.0
for Y in years:
    rows_y = np.where((d["date"].dt.year == Y).values & valid)[0]
    if len(rows_y) == 0:
        continue
    cut = rows_y[0] - GAP
    tr_idx = np.arange(first_valid, cut + 1)
    tr_idx = tr_idx[valid[tr_idx] & ~np.isnan(R2[tr_idx])]
    net, Z = train_year(tr_idx, SEED * 100 + Y)
    with torch.no_grad():
        for t in rows_y:
            s = torch.cat([torch.from_numpy(Z[t:t + 1]), torch.tensor([[prev]])], 1)
            a = float(torch.sigmoid(net.pi(s)).item())
            if MODE == "overlay":
                a = float(np.clip(np.nan_to_num(BASEW[t]) * 2 * a, 0, 1))
            w[t] = a
            prev = a
    print(f"kappa={KAPPA} seed={SEED} year={Y} trained on {len(tr_idx)} rows", flush=True)

pd.to_pickle(w, RES / f"step4_rl_{MODE}_k{KAPPA}_s{SEED}.pkl")
m = evaluate(d, pd.Series(w, index=d.index), "dev")
print("DEV", MODE, KAPPA, SEED, {k: round(v, 4) for k, v in m.items()})
