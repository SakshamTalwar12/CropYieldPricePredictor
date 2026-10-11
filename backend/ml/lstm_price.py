"""Optional LSTM price model (the "XGBoost/LSTM" branch of Stage 2).

The LSTM reads the last L years of the crop-region price/yield history (known past
only) and combines the final hidden state with static inputs - one-hot crop/region/
season and the Stage-1 predicted yield - to predict the log-price change relative to
the most recent historical price. Because it predicts a change from the last known
price it can be rolled forward to years beyond the training data.

Requires PyTorch (optional dependency; the rest of the project works without it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    TORCH_OK = True
except ImportError:           # pragma: no cover
    TORCH_OK = False

from backend.config import SEED

SEQ_LEN = 5
CATS = ["crop", "region", "season"]


def build_history(df: pd.DataFrame) -> dict:
    """{(crop, region): DataFrame[year, logp, yield_ratio]} from rows of df."""
    crop_yield = df.groupby("crop")["yield_qha"].mean()
    g = (df.assign(logp=np.log(df["price_rs_per_quintal"]),
                   yr=df["yield_qha"] / df["crop"].map(crop_yield))
           .groupby(["crop", "region", "year"], as_index=False)
           .agg(logp=("logp", "mean"), yield_ratio=("yr", "mean")))
    hist = {k: v.sort_values("year").reset_index(drop=True)
            for k, v in g.groupby(["crop", "region"])}
    return {"series": hist, "crop_yield": crop_yield.to_dict()}


def _window(hist, crop, region, year):
    """Return (seq[L,2], last_logp, gap) or None if no history before `year`."""
    h = hist["series"].get((crop, region))
    if h is None:
        return None
    h = h[h["year"] < year].tail(SEQ_LEN)
    if h.empty:
        return None
    last = float(h["logp"].iloc[-1])
    seq = np.stack([h["logp"].values - last, h["yield_ratio"].values - 1.0], axis=1)
    if len(seq) < SEQ_LEN:                       # left-pad by repeating the oldest step
        seq = np.vstack([np.repeat(seq[:1], SEQ_LEN - len(seq), axis=0), seq])
    gap = float(year - int(h["year"].iloc[-1]) - 1)
    return seq.astype(np.float32), last, gap


if TORCH_OK:
    class Net(nn.Module):
        def __init__(self, n_static, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(2, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(),
                                      nn.Linear(64, 1))

        def forward(self, seq, static):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], static], 1)).squeeze(1)


class LSTMPrice:
    def __init__(self, epochs=25, batch=256, lr=2e-3):
        self.epochs, self.batch, self.lr = epochs, batch, lr

    # --- feature assembly
    def _static(self, df, yhat, gap):
        cats = pd.get_dummies(df[CATS], dtype=float).reindex(columns=self.cat_cols, fill_value=0.0)
        yr = np.asarray(yhat, float) / df["crop"].map(self.hist["crop_yield"]).values - 1.0
        return np.column_stack([cats.values, yr, np.asarray(gap, float) / 3.0]).astype(np.float32)

    def _prepare(self, df, yhat):
        seqs, last, gaps, keep = [], [], [], []
        for i, (c, r, y) in enumerate(zip(df["crop"], df["region"], df["year"])):
            w = _window(self.hist, c, r, int(y))
            if w is None:
                continue
            seqs.append(w[0]); last.append(w[1]); gaps.append(w[2]); keep.append(i)
        keep = np.array(keep, int)
        sub = df.iloc[keep]
        return (np.stack(seqs), self._static(sub, np.asarray(yhat)[keep], gaps),
                np.array(last), keep)

    def fit(self, df: pd.DataFrame, yhat):
        torch.manual_seed(SEED); np.random.seed(SEED)
        self.hist = build_history(df)
        self.cat_cols = list(pd.get_dummies(df[CATS], dtype=float).columns)
        seq, static, last, keep = self._prepare(df, yhat)
        target = (np.log(df["price_rs_per_quintal"].values[keep]) - last).astype(np.float32)
        self.net = Net(static.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        S, T, Y = map(torch.from_numpy, (seq, static, target))
        n = len(Y)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(S[b], T[b]), Y[b])
                loss.backward(); opt.step()
        return self

    def predict(self, df: pd.DataFrame, yhat) -> np.ndarray:
        """Price per row; NaN where no history exists for the crop-region."""
        out = np.full(len(df), np.nan)
        seq, static, last, keep = self._prepare(df, yhat)
        if len(keep):
            self.net.eval()
            with torch.no_grad():
                d = self.net(torch.from_numpy(seq), torch.from_numpy(static)).numpy()
            out[keep] = np.exp(last + d)
        return out

    # --- persistence
    def state(self) -> dict:
        return {"net": self.net.state_dict(), "hist": self.hist, "cat_cols": self.cat_cols,
                "n_static": len(self.cat_cols) + 2}

    @classmethod
    def from_state(cls, st) -> "LSTMPrice":
        m = cls(); m.hist, m.cat_cols = st["hist"], st["cat_cols"]
        m.net = Net(st["n_static"]); m.net.load_state_dict(st["net"]); m.net.eval()
        return m
