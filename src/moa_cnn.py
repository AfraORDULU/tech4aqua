"""
Tech4Aqua - Tablo verisi icin "soft-ordering" 1D-CNN (orijinal mimari).

Kaynak: Kaggle "Mechanisms of Action (MoA) Prediction" yarismasi 2. sira
cozumu, baosenguo. https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution
(commit bda770cdd2c458c23c006883dc8364066289c63c, training/1d-cnn-train.ipynb)

`_MoAModel` sinifi orijinal `Model` sinifinin birebir kopyasidir
(hidden_size=4096, kanal 256/512/512, weight_norm, BatchNorm, Dropout,
CELU, kisa yol carpimi). Tek degisiklik: cikis sayisi 1 (ikili sinif).
Fikir: ozellikler once 4096 boyutlu bir katmana genisletilir ve 256 kanal x
16 uzunlukta bir "sinyale" donusturulur; evrisim bu ogrenilmis siralama
uzerinde calisir, bu yuzden sutunlarin keyfi sirasi sorun olmaz.

Orijinal egitim ayarlari (ana egitim adimi) aynen kullanilir:
  QuantileTransformer(n_quantiles=100, output_distribution="normal"),
  Adam(lr=1e-3, weight_decay=1e-5), OneCycleLR(pct_start=0.1,
  div_factor=1e3, max_lr=1e-2), 25 tur, batch 128,
  SmoothBCEwLogits(smoothing=0.001).
Bu veriye uygulanamadigi icin alinmayan adimlar (raporda belirtilir):
  * PCA ek ozellikleri (gen/hucre sutunlarina ozgu; bizde 5-11 ozellik var),
  * skorlanmayan hedeflerle 1 turluk on egitim (bizde ek hedef yok),
  * en iyi turun dogrulama katina gore secilmesi: bizde dogrulama kati test
    katidir, bu sizinti olurdu; son turun modeli kullanilir.
Eksik deger doldurma (medyan) yalnizca egitim verisiyle yapilir.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from torch import optim

warnings.filterwarnings("ignore", message=".*weight_norm.*")
torch.set_num_threads(2)


class _MoAModel(nn.Module):
    # --- orijinal koddan birebir (num_targets=1) ---
    def __init__(self, num_features, num_targets, hidden_size):
        super(_MoAModel, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False), dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


def _smooth_bce(logits, targets, smoothing=0.001):
    # orijinal SmoothBCEwLogits (agirliksiz): y * (1 - s) + 0.5 * s
    targets = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, targets)


class MoACNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=0, epochs=25, batch_size=128, hidden_size=4096):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            self.imputer_ = SimpleImputer(strategy="median").fit(X)
            Xi = self.imputer_.transform(X)
            self.qt_ = QuantileTransformer(
                n_quantiles=min(100, len(Xi)), random_state=0,
                output_distribution="normal").fit(Xi)
        X = self.qt_.transform(self.imputer_.transform(X))
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x = self._prep(X, fit=True)
        yt = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        self.classes_ = np.array([0, 1])
        self.net_ = _MoAModel(x.shape[1], 1, self.hidden_size)
        n = len(x)
        # BatchNorm tek ornekli son grupla calismaz: o tek ornek atlanir
        steps = int(np.ceil(n / self.batch_size))
        if n % self.batch_size == 1:
            steps -= 1
        opt = torch.optim.Adam(self.net_.parameters(), lr=1e-3,
                               weight_decay=1e-5)
        sch = optim.lr_scheduler.OneCycleLR(
            optimizer=opt, pct_start=0.1, div_factor=1e3, max_lr=1e-2,
            epochs=self.epochs, steps_per_epoch=steps)
        g = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(steps):
                idx = perm[s * self.batch_size:(s + 1) * self.batch_size]
                opt.zero_grad()
                loss = _smooth_bce(self.net_(x[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
                sch.step()
        self.net_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net_(self._prep(X)).squeeze(-1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
