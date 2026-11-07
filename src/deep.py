"""
Tech4Aqua - Derin ogrenme ve alan genelleme (domain generalization) modelleri.

Uc yontem ayni kucuk ag (MLP) uzerinde calisir; farklari kayip fonksiyonudur:

  ERM      : Siradan egitim. Tum numunelerin agirlikli BCE kaybi.
  GroupDRO : Gruplar (belediye x sinif) arasinda EN KOTU grubun kaybini
             azaltmaya calisir (Sagawa vd., 2020). Amac: egitimde zayif
             temsil edilen bolgelerde de calisan bir model.
  CORAL    : Gizli katmandaki ozelliklerin kovaryansini belediyeler
             arasinda hizalar (Sun & Saenko, 2016). Amac: bolgeden
             bagimsiz bir temsil ogrenmek.

Alan (domain) olarak belediye (Chaves, Montalegre, Boticas) kullanilir.
ZAPE'ler cok sayida ve cogu az numuneli oldugu icin dogrudan alan olarak
kullanilmaya uygun degildir.

Model scikit-learn arayuzune uyar: fit(X, y, groups) / predict_proba(X).
Eksik deger doldurma ve olcekleme modelin icinde, yalnizca egitim
verisiyle yapilir.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

torch.set_num_threads(2)


class _MLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(z).squeeze(-1), z


def _coral(z_a, z_b):
    d = z_a.shape[1]
    ca = torch.cov(z_a.T)
    cb = torch.cov(z_b.T)
    return ((ca - cb) ** 2).sum() / (4 * d * d)


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    needs_groups = True

    def __init__(self, method="erm", hidden=(32, 16), dropout=0.2,
                 epochs=300, lr=1e-3, weight_decay=1e-3, dro_eta=0.1,
                 coral_weight=1.0, seed=0):
        self.method = method
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dro_eta = dro_eta
        self.coral_weight = coral_weight
        self.seed = seed

    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            self.imputer_ = SimpleImputer(strategy="median").fit(X)
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        return torch.tensor(
            self.scaler_.transform(self.imputer_.transform(X)),
            dtype=torch.float32)

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x = self._prep(X, fit=True)
        y_np = np.asarray(y, dtype=float)
        yt = torch.tensor(y_np, dtype=torch.float32)
        dom = np.zeros(len(y_np), int) if groups is None else \
            np.unique(np.asarray(groups), return_inverse=True)[1]

        pos = max(y_np.sum(), 1.0)
        pos_weight = torch.tensor((len(y_np) - pos) / pos)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        self.net_ = _MLP(x.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        # GroupDRO gruplari: alan x sinif
        g_id = dom * 2 + y_np.astype(int)
        g_list = [torch.tensor(g_id == g) for g in np.unique(g_id)]
        q = torch.ones(len(g_list)) / len(g_list)
        d_list = [torch.tensor(dom == d) for d in np.unique(dom)
                  if (dom == d).sum() > 2]

        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logit, z = self.net_(x)
            losses = bce(logit, yt)
            if self.method == "groupdro":
                g_loss = torch.stack([losses[m].mean() for m in g_list])
                with torch.no_grad():
                    q = q * torch.exp(self.dro_eta * g_loss)
                    q = q / q.sum()
                loss = (q * g_loss).sum()
            else:
                loss = losses.mean()
                if self.method == "coral" and len(d_list) > 1:
                    pen = 0.0
                    for i in range(len(d_list)):
                        for j in range(i + 1, len(d_list)):
                            pen = pen + _coral(z[d_list[i]], z[d_list[j]])
                    loss = loss + self.coral_weight * pen
            loss.backward()
            opt.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logit, _ = self.net_(self._prep(X))
            p = torch.sigmoid(logit).numpy()
        return np.column_stack([1 - p, p])
