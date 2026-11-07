"""
Tech4Aqua - Deney calistirma, esik secimi ve metrikler.

run_cv          : Tekrarli ZAPE gruplu (arindirilmis) CV ile her numune
                  icin test-disi (out-of-fold) olasilik uretir.
choose_threshold: Karar esigini YALNIZCA egitim verisi icinde, ic gruplu
                  CV ile secer (MCC'yi en buyuten esik). Test verisi esik
                  seciminde hic kullanilmaz.
summarize       : Tohum (seed) basina metrik hesaplar, ortalama/std verir.
zape_bootstrap  : ZAPE duzeyinde yeniden ornekleme ile %95 guven araligi.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score, brier_score_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from src.models import make_model
from src.validation import grouped_splits


# ---------------------------------------------------------------- metrikler
def compute_metrics(y, prob, pred=None):
    y = np.asarray(y)
    prob = np.asarray(prob)
    out = {
        "n": len(y),
        "n_pos": int(y.sum()),
        "prevalence": y.mean(),
        "ROC_AUC": roc_auc_score(y, prob) if 0 < y.sum() < len(y) else np.nan,
        "PR_AUC": average_precision_score(y, prob) if y.sum() > 0 else np.nan,
    }
    if pred is not None:
        pred = np.asarray(pred)
        out["MCC"] = matthews_corrcoef(y, pred)
        out["precision"] = precision_score(y, pred, zero_division=0)
        out["recall"] = recall_score(y, pred, zero_division=0)
        out["flag_rate"] = pred.mean()
    if np.all((prob >= 0) & (prob <= 1)):
        out["brier"] = brier_score_loss(y, prob)
    return out


def best_mcc_threshold(y, prob):
    """MCC'yi en buyuten esik (aday esikler: benzersiz olasiliklar)."""
    candidates = np.unique(np.quantile(prob, np.linspace(0.02, 0.98, 97)))
    scores = [matthews_corrcoef(y, (prob >= t).astype(int))
              for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def fit_model(model, df, features, idx):
    """Modeli egitir. Alan genelleme modellerine belediye bilgisi verilir."""
    X, y = df.iloc[idx][features], df["y"].values[idx]
    if getattr(model, "needs_groups", False):
        return model.fit(X, y, groups=df["municipality"].values[idx])
    return model.fit(X, y)


# ------------------------------------------------------------- esik secimi
def choose_threshold(df, features, model_name, train_idx, seed,
                     inner_splits=4, purge=True):
    """Egitim katinin icinde ic CV ile esik secer.

    Egitimde cok az ZAPE varsa ic bolme sayisi azaltilir; ic CV
    yapilamiyorsa 0.5 dondurulur.
    """
    n_groups = len(set(df["zape"].values[train_idx]))
    inner_splits = min(inner_splits, n_groups)
    if inner_splits < 2:
        return 0.5
    inner_prob = np.full(len(df), np.nan)
    for tr, te in grouped_splits(df, inner_splits, seed + 1000, purge,
                                 index=train_idx):
        m = fit_model(make_model(model_name, seed), df, features, tr)
        inner_prob[te] = m.predict_proba(df.iloc[te][features])[:, 1]
    mask = ~np.isnan(inner_prob)
    return best_mcc_threshold(df["y"].values[mask], inner_prob[mask])


# ------------------------------------------------------------ deney dongusu
def run_cv(df, features, model_name, seeds=(0, 1, 2, 3, 4), n_splits=5,
           purge=True, train_mask=None, tune_threshold=True):
    """Tekrarli gruplu CV. Her (seed, satir) icin olasilik ve karar.

    train_mask: verilirse egitim yalnizca bu satirlarla yapilir
                (or. yalniz rutin numunelerle egitim). Test her zaman
                tum satirlarda yapilir; alt kume raporu summarize'da.
    """
    rows = []
    for seed in seeds:
        for fold, (tr, te) in enumerate(
                grouped_splits(df, n_splits, seed, purge)):
            if train_mask is not None:
                tr = tr[np.asarray(train_mask)[tr]]
            model = fit_model(make_model(model_name, seed), df, features,
                              tr)
            prob = model.predict_proba(df.iloc[te][features])[:, 1]
            thr = (choose_threshold(df, features, model_name, tr, seed,
                                    purge=purge)
                   if tune_threshold else 0.5)
            rows.append(pd.DataFrame({
                "seed": seed, "fold": fold, "row": te,
                "prob": prob, "threshold": thr,
                "pred": (prob >= thr).astype(int),
            }))
    return pd.concat(rows, ignore_index=True)


def run_temporal(df, features, model_name, splits, seeds=(0, 1, 2)):
    """Zamansal bolmeler icin test-disi olasiliklar.

    splits: (etiket, train_idx, test_idx) listesi
            (or. validation.temporal_splits ciktisi).
    Esik yine yalnizca egitim verisi icinde secilir.
    """
    rows = []
    for seed in seeds:
        for label, tr, te in splits:
            model = fit_model(make_model(model_name, seed), df, features,
                              tr)
            prob = model.predict_proba(df.iloc[te][features])[:, 1]
            thr = choose_threshold(df, features, model_name, tr, seed)
            rows.append(pd.DataFrame({
                "seed": seed, "fold": label, "row": te, "prob": prob,
                "threshold": thr, "pred": (prob >= thr).astype(int),
            }))
    return pd.concat(rows, ignore_index=True)


def summarize(df, oof, subset=None):
    """Tohum basina metrikler. subset: satirlar icin boolean maske."""
    per_seed = []
    for seed, g in oof.groupby("seed"):
        if subset is not None:
            g = g[np.asarray(subset)[g["row"].values]]
        y = df["y"].values[g["row"].values]
        m = compute_metrics(y, g["prob"].values, g["pred"].values)
        # Sabit 0.5 esigi (class_weight="balanced" modeller icin dogal esik)
        m["MCC_05"] = matthews_corrcoef(y, (g["prob"].values >= 0.5)
                                        .astype(int))
        m["threshold_mean"] = g["threshold"].mean()
        m["seed"] = seed
        per_seed.append(m)
    per_seed = pd.DataFrame(per_seed)
    return per_seed


def mean_std(per_seed, cols=("ROC_AUC", "PR_AUC", "MCC", "MCC_05",
                             "precision", "recall", "brier",
                             "threshold_mean")):
    out = {}
    for c in cols:
        if c in per_seed:
            out[c] = per_seed[c].mean()
            out[c + "_std"] = per_seed[c].std(ddof=0)
    for c in ("n", "n_pos", "prevalence"):
        out[c] = per_seed[c].iloc[0]
    return out


def mean_prob(oof):
    """Tohumlar uzerinden ortalama olasilik (bootstrap icin)."""
    return oof.groupby("row")["prob"].mean()


# --------------------------------------------------------------- bootstrap
def zape_bootstrap(df, rows, score_a, score_b=None, metric="ROC_AUC",
                   n_boot=2000, seed=0):
    """ZAPE duzeyinde bootstrap.

    score_b verilirse fark (a - b) icin guven araligi dondurur
    (eslestirilmis test: ayni yeniden orneklem iki skora uygulanir).
    """
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    y = df["y"].values[rows]
    zapes = df["zape"].values[rows]
    uniq = np.unique(zapes)
    by_zape = {z: np.where(zapes == z)[0] for z in uniq}
    fn = roc_auc_score if metric == "ROC_AUC" else average_precision_score

    def stat(idx):
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            return np.nan
        a = fn(y[idx], score_a[idx])
        return a if score_b is None else a - fn(y[idx], score_b[idx])

    point = stat(np.arange(len(rows)))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_zape[z] for z in pick])
        boots.append(stat(idx))
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = (boots <= 0).mean() if score_b is not None else np.nan
    return {"point": point, "ci_low": lo, "ci_high": hi,
            "p_one_sided": p}
