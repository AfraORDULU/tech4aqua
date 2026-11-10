"""
Tech4Aqua - Bolge gecmisini DIZI olarak okuyan derin modeller (GRU, 1D-CNN).

E09'da bolge gecmisi elle secilmis ozet sayilarla verildi. Burada model,
ayni ZAPE'deki onceki numunelerin kendisini zaman sirasiyla okur ve neyin
onemli olduguna kendisi karar verir.

Sizinti kurali (src/history.py ile ayni): Bir numune icin yalnizca ayni
ZAPE'de, numune gununden en az LAB_DELAY_DAYS gun once alinmis (sonucu karar
aninda bilinen) numuneler kullanilir. Gecmisi olmayan numunede dizi bostur;
model yalnizca bugunku olcumlere dayanir.

Dizinin her adiminda (en fazla MAX_LEN adim, en yenisi sonda):
  klor, maks. sicaklik, yagis, nem, sin(ay), cos(ay), sonuc (0/1),
  log(1 + o numuneden bugune gecen gun)
Bugunku numune: A seti ozellikleri + log(1 + bilinen gecmis sayisi)

Model scikit-learn arayuzune uyar. Girdi tablosunda "_row" sutunu
(load_dataset satir numarasi) bulunur; diziler bu numarayla bulunur.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from src.data import CHLORINE, FEATURE_SETS
from src.history import LAB_DELAY_DAYS

torch.set_num_threads(2)

MAX_LEN = 16
STORE = None  # build_store(df) ile bir kez doldurulur
STORE_HYB = None  # build_store(df, CURRENT + gecmis ozetleri)
STEP_RAW = [CHLORINE, "TEMP MÁX (ºC)", "PRECIPITAÇÃO (mm)",
            "UMIDADE MÉDIA (%)"]
CURRENT = FEATURE_SETS["A"]


def build_sequences(df, lab_delay_days=LAB_DELAY_DAYS, max_len=MAX_LEN):
    """Her satir icin (gecmis indeksleri, gecen gunler) listesi."""
    hist_idx = [[] for _ in range(len(df))]
    hist_gap = [[] for _ in range(len(df))]
    for _, g in df.groupby("zape"):
        g = g.sort_values("date")
        idx = g.index.values
        dates = g["date"].values
        for i, d in zip(idx, dates):
            known = dates <= d - np.timedelta64(lab_delay_days, "D")
            k_idx = idx[known][-max_len:]
            k_gap = ((d - dates[known][-max_len:]) /
                     np.timedelta64(1, "D")).astype(float)
            hist_idx[i] = list(k_idx)
            hist_gap[i] = list(k_gap)
    return hist_idx, hist_gap


class _Net(nn.Module):
    def __init__(self, kind, n_step, n_cur, hidden=16, dropout=0.2):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.enc = nn.GRU(n_step, hidden, batch_first=True)
        else:  # cnn
            self.enc = nn.Sequential(
                nn.Conv1d(n_step, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + n_cur, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, seq, mask, cur):
        # seq: (B, L, F) sola dayali dolgu; mask: (B, L) 1 = gercek adim
        has = (mask.sum(1, keepdim=True) > 0).float()
        if self.kind == "gru":
            out, _ = self.enc(seq)                     # (B, L, H)
            # son gercek adim her zaman L-1 (sola dolgu), bos dizide 0
            h = out[:, -1, :] * has
        else:
            z = self.enc(seq.transpose(1, 2))          # (B, H, L)
            z = z.masked_fill(mask.unsqueeze(1) == 0, -1e4)
            h = z.max(dim=2).values * has
        return self.head(torch.cat([h, cur], dim=1)).squeeze(-1)


class SequenceClassifier(BaseEstimator, ClassifierMixin):
    """kind: 'gru' veya 'cnn'. store: build_store(df) ciktisi."""

    def __init__(self, kind="gru", store=None, epochs=200, lr=3e-3,
                 weight_decay=1e-3, max_len=MAX_LEN, seed=0):
        self.kind = kind
        self.max_len = max_len
        self.store = store
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    # ------------------------------------------------------------ tensors
    def _tensors(self, rows):
        s = self.store
        L = self.max_len
        n_step = s["step"].shape[1] + 2
        seq = np.zeros((len(rows), L, n_step), dtype=np.float32)
        mask = np.zeros((len(rows), L), dtype=np.float32)
        for b, r in enumerate(rows):
            hi, hg = s["hist_idx"][r][-L:], s["hist_gap"][r][-L:]
            n = len(hi)
            if n == 0:
                continue
            st = (s["step"][hi] - self.step_mu_) / self.step_sd_
            st = np.nan_to_num(st, nan=0.0)
            feat = np.column_stack([st, s["y"][hi],
                                    np.log1p(hg) / 5.0])
            seq[b, L - n:, :] = feat
            mask[b, L - n:] = 1
        cur = s["cur"][rows]
        cur = np.where(np.isnan(cur), self.cur_med_, cur)
        cur = (cur - self.cur_mu_) / self.cur_sd_
        nh = np.log1p([len(s["hist_idx"][r]) for r in rows])[:, None] / 3.0
        # not: bilinen gecmis sayisi tum gecmisten (en fazla MAX_LEN) sayilir
        cur = np.hstack([cur, nh]).astype(np.float32)
        return (torch.tensor(seq), torch.tensor(mask), torch.tensor(cur))

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rows = X["_row"].values.astype(int)
        s = self.store
        st = s["step"][rows]
        self.step_mu_ = np.nanmean(st, 0)
        self.step_sd_ = np.nanstd(st, 0) + 1e-6
        cur = s["cur"][rows]
        self.cur_med_ = np.nanmedian(cur, 0)
        cur = np.where(np.isnan(cur), self.cur_med_, cur)
        self.cur_mu_ = cur.mean(0)
        self.cur_sd_ = cur.std(0) + 1e-6
        seq, mask, curt = self._tensors(rows)
        yt = torch.tensor(np.asarray(y, dtype=np.float32))
        pos = max(float(yt.sum()), 1.0)
        bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((len(yt) - pos) / pos))
        self.net_ = _Net(self.kind, seq.shape[2], curt.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = bce(self.net_(seq, mask, curt), yt)
            loss.backward()
            opt.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        rows = X["_row"].values.astype(int)
        seq, mask, cur = self._tensors(rows)
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(seq, mask, cur)).numpy()
        return np.column_stack([1 - p, p])


def build_store(df, current_cols=None):
    """Bir kez hesaplanan dizi deposu (tum satirlar icin).

    current_cols: bugunku numune icin kullanilacak sutunlar (varsayilan A
    seti). Hibrit model icin A seti + src.history ozetleri verilir.
    """
    current_cols = CURRENT if current_cols is None else current_cols
    hist_idx, hist_gap = build_sequences(df)
    m = df["date"].dt.month.values
    step = np.column_stack([df[c].values.astype(float) for c in STEP_RAW] +
                           [np.sin(2 * np.pi * m / 12),
                            np.cos(2 * np.pi * m / 12)])
    return {"hist_idx": hist_idx, "hist_gap": hist_gap, "step": step,
            "zape": df["zape"].values,
            "y": df["y"].values.astype(float),
            "cur": df[current_cols].values.astype(float)}


class SelectedSequenceClassifier(BaseEstimator, ClassifierMixin):
    """Ayarlari YALNIZCA egitim verisi icinde secen dizi modeli.

    fit() icinde egitim satirlari ZAPE'ye gore 3 gruba bolunur; her aday
    ayar (dizi uzunlugu x epoch) icin ic dogrulamada ortalama hassasiyet
    (PR-AUC) hesaplanir; en iyi ayarla tum egitim verisi yeniden egitilir.
    Test verisi ayar seciminde hic kullanilmaz.
    """

    GRID = [(4, 100), (4, 200), (8, 100), (8, 200), (16, 100), (16, 200)]

    def __init__(self, kind="gru", store=None, seed=0, inner_splits=3):
        self.kind = kind
        self.store = store
        self.seed = seed
        self.inner_splits = inner_splits

    def fit(self, X, y, groups=None):
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import GroupKFold
        y = np.asarray(y)
        zapes = self.store["zape"][X["_row"].values.astype(int)]
        scores = {}
        n_groups = len(set(zapes))
        k = min(self.inner_splits, n_groups)
        for L, ep in self.GRID:
            preds = np.full(len(y), np.nan)
            if k >= 2:
                for tr, te in GroupKFold(n_splits=k).split(X, y, zapes):
                    if y[tr].sum() == 0:
                        continue
                    m = SequenceClassifier(self.kind, self.store, epochs=ep,
                                           max_len=L, seed=self.seed)
                    m.fit(X.iloc[tr], y[tr])
                    preds[te] = m.predict_proba(X.iloc[te])[:, 1]
            ok = ~np.isnan(preds)
            scores[(L, ep)] = (average_precision_score(y[ok], preds[ok])
                               if ok.sum() and 0 < y[ok].sum() < ok.sum()
                               else -1)
        self.best_ = max(scores, key=scores.get)
        self.scores_ = scores
        L, ep = self.best_
        self.model_ = SequenceClassifier(self.kind, self.store, epochs=ep,
                                         max_len=L, seed=self.seed)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
