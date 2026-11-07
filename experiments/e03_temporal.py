"""
Deney 3 - Zamana gore dogrulama (hocanin 3. maddesi).

Gercek hayatta model gecmisle egitilip gelecegi tahmin edecek.
Bu yuzden egitim verisi her zaman test verisinden ONCEKI tarihlerden olusur.

Protokoller
  rolling     : Her test yili Y icin, Y'den onceki tum yillarla egit,
                Y yilini test et (genisleyen pencere). Yillarin tahminleri
                birlestirilip tek bir ROC/PR hesaplanir.
  holdout2020 : 2008-2019 ile egit, 2020-2025'i test et.
  Her ikisi iki halde: unseen_zape=False (yalniz gelecek) ve
  unseen_zape=True (hem gelecek hem gorulmemis bolge).

Ayrica: 2025 cikarilinca ZAPE gruplu CV sonucunun ne kadar dustugu.

Sonuclar hem tum numunelerde hem yalniz rutin numunelerde raporlanir.
Yalniz klor siralamasi da ayni test numunelerinde hesaplanir.

Calistirma: python -m experiments.e03_temporal
Ciktilar: results/e03_temporal/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_std, run_cv, run_temporal,
                            summarize)
from src.models import MODEL_NAMES
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e03_temporal"
SEEDS = (0, 1, 2)
TEST_YEARS = range(2015, 2026)


def holdout_split(df, cut_year, unseen_zape):
    te = np.where(df["year"].values >= cut_year)[0]
    tr_mask = df["year"].values < cut_year
    if unseen_zape:
        tr_mask &= ~df["zape"].isin(set(df["zape"].values[te])).values
    return [(f"<{cut_year}", np.where(tr_mask)[0], te)]


def chlorine_row(df, rows_idx, subset):
    idx = rows_idx[subset[rows_idx]]
    cl = df[CHLORINE]
    score = -cl.fillna(cl.median()).values
    m = compute_metrics(df["y"].values[idx], score[idx])
    m.pop("brier", None)
    return m


def evaluate(df, oof, name, extra):
    routine = ~df["is_investigation"].values
    no2025 = df["year"].values < 2025
    rows = []
    test_rows = np.unique(oof["row"].values)
    for subset_name, subset in (("Tumu", np.ones(len(df), bool)),
                                ("Tumu_2025_haric", no2025),
                                ("Rutin", routine)):  # 2025'te rutin yok
        if df["y"].values[test_rows[subset[test_rows]]].sum() == 0:
            continue
        m = mean_std(summarize(df, oof, subset))
        rows.append({**extra, "model": name, "test": subset_name, **m})
        if name == "LR":  # klor satiri bir kez yeter
            c = chlorine_row(df, test_rows, subset)
            rows.append({**extra, "model": "YALNIZ_KLOR", "test": subset_name,
                         **c})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    results, per_year, oofs = [], [], []

    for protocol in ("rolling", "holdout2020"):
        for unseen in (False, True):
            if protocol == "rolling":
                splits = list(temporal_splits(df, TEST_YEARS, unseen))
            else:
                splits = holdout_split(df, 2020, unseen)
            for fs in ("A", "B", "C"):
                for model in MODEL_NAMES:
                    oof = run_temporal(df, FEATURE_SETS[fs], model, splits,
                                       SEEDS)
                    extra = {"protocol": protocol, "unseen_zape": unseen,
                             "feature_set": fs}
                    rows = evaluate(df, oof, model, extra)
                    if fs != "A":  # klor satiri yalniz A blogunda tutulur
                        rows = [r for r in rows
                                if r["model"] != "YALNIZ_KLOR"]
                    results += rows
                    oofs.append(oof.assign(**extra, model=model))
                    r0 = rows[0]
                    print(f"{protocol:11s} unseen={unseen!s:5s} {fs} "
                          f"{model:4s} n={r0['n']} pos={r0['n_pos']} "
                          f"ROC={r0['ROC_AUC']:.3f} PR={r0['PR_AUC']:.3f}")
                    # yil bazinda (yalniz rolling, LR)
                    if protocol == "rolling" and model == "LR":
                        first = oof[oof["seed"] == SEEDS[0]]
                        for year, g in first.groupby("fold"):
                            idx = g["row"].values
                            y = df["y"].values[idx]
                            m = compute_metrics(y, g["prob"].values)
                            per_year.append({**extra, "model": model,
                                             "test_year": year,
                                             **m})

    # 2025 cikarilinca ZAPE gruplu CV
    sub = df[df["year"] < 2025].reset_index(drop=True)
    for fs in ("A", "C"):
        for model in MODEL_NAMES:
            oof = run_cv(sub, FEATURE_SETS[fs], model, (0, 1, 2, 3, 4))
            m = mean_std(summarize(sub, oof))
            results.append({"protocol": "zape_cv_2025_haric",
                            "unseen_zape": True, "feature_set": fs,
                            "model": model, "test": "Tumu", **m})
            print(f"zape_cv_2025_haric {fs} {model}: "
                  f"ROC={m['ROC_AUC']:.3f} PR={m['PR_AUC']:.3f}")

    pd.DataFrame(results).round(4).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_year).round(4).to_csv(OUT / "per_year_LR.csv",
                                           index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
