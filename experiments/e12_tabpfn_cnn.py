"""
Deney 12 - TabPFN ve 1D-CNN (tablo verisi uzerinde), danisman onerisi.

  TABPFN : Tablo verisi icin onceden egitilmis transformer (TabPFN v2,
           Hollmann ve dig., 2025). Egitim verisini "baglam" olarak alip
           tahmin yapar; hiperparametre ayari gerektirmez.
  CNN1D  : Ozellik vektorunu tek kanalli 1 boyutlu sinyal gibi okuyan
           evrisimli ag (src/deep.py).
Karsilastirma: LR (ana model) ve egitimsiz klor. Ozellik seti: A.
Protokoller E05 ile ayni: ZAPE gruplu arindirilmis CV, genisleyen pencere
(gorulmemis ZAPE), belediye disarida birakma.

TabPFN model dosyasi: TABPFN_MODEL_PATH ortam degiskeni ile verilir
(or. tabpfn-v2-classifier.ckpt). Dosya yoksa TabPFN atlanir.

Calistirma: TABPFN_MODEL_PATH=/yol/tabpfn-v2-classifier.ckpt \
            python -m experiments.e12_tabpfn_cnn
Ciktilar: results/e12_tabpfn_cnn/
"""

import os

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e12_tabpfn_cnn"
SEEDS = (0, 1, 2)
A = FEATURE_SETS["A"]


def lomo_splits(df):
    towns = df["municipality"].values
    for town in np.unique(towns):
        yield town, np.where(towns != town)[0], np.where(towns == town)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    models = ["LR", "CNN1D"]
    if os.path.exists(os.environ.get("TABPFN_MODEL_PATH", "")):
        models.append("TABPFN")
    else:
        print("TABPFN_MODEL_PATH bulunamadi: TabPFN atlaniyor")

    routine = ~df["is_investigation"].values
    no2025 = df["year"].values < 2025
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine,
               "Tumu_2025_haric": no2025}
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values
    y = df["y"].values
    protocols = {
        "zape_cv": None,
        "rolling_unseen": list(temporal_splits(df, range(2015, 2026),
                                               unseen_zape=True)),
        "lomo": list(lomo_splits(df)),
    }
    rows, probs = [], {}
    for protocol, splits in protocols.items():
        test_rows = None
        for model in models:
            oof = (run_cv(df, A, model, SEEDS) if splits is None
                   else run_temporal(df, A, model, splits, SEEDS))
            probs[(protocol, model)] = mean_prob(oof)
            test_rows = np.unique(oof["row"].values)
            line = []
            for sname, smask in subsets.items():
                if y[test_rows[smask[test_rows]]].sum() < 3:
                    continue
                m = mean_std(summarize(df, oof, smask))
                rows.append({"protocol": protocol, "model": model,
                             "test": sname, **m})
                line.append(f"{sname}: ROC={m['ROC_AUC']:.3f} "
                            f"PR={m['PR_AUC']:.3f}")
            print(f"{protocol:14s} {model:6s} " + " | ".join(line))
        for sname, smask in subsets.items():
            idx = test_rows[smask[test_rows]]
            if y[idx].sum() < 3:
                continue
            m = compute_metrics(y[idx], chlorine[idx])
            m.pop("brier", None)
            rows.append({"protocol": protocol, "model": "YALNIZ_KLOR",
                         "test": sname, **m})
    pd.DataFrame(rows).round(4).to_csv(OUT / "summary.csv", index=False)

    # Belediye bazinda (LOMO)
    lomo = []
    for town, _, te in protocols["lomo"]:
        if y[te].sum() == 0:
            continue
        for model in models:
            p = probs[("lomo", model)].reindex(te).values
            lomo.append({"test_belediye": town, "model": model,
                         **compute_metrics(y[te], p)})
        lomo.append({"test_belediye": town, "model": "YALNIZ_KLOR",
                     **compute_metrics(y[te], chlorine[te])})
    pd.DataFrame(lomo).round(4).to_csv(OUT / "lomo_by_town.csv", index=False)

    # Eslestirilmis ZAPE bootstrap: her model - LR ve - klor (ZAPE CV)
    tests = []
    lr = probs[("zape_cv", "LR")].sort_index().values
    for model in models[1:]:
        p = probs[("zape_cv", model)].sort_index().values
        for ref_name, ref in (("LR", lr), ("YALNIZ_KLOR", chlorine)):
            for sname in ("Tumu", "Rutin"):
                idx = np.where(subsets[sname])[0]
                for metric in ("ROC_AUC", "PR_AUC"):
                    b = zape_bootstrap(df, idx, p[idx], ref[idx],
                                       metric=metric)
                    tests.append({"karsilastirma": f"{model} - {ref_name}",
                                  "test": sname, "metrik": metric, **b})
    pd.DataFrame(tests).round(4).to_csv(OUT / "bootstrap.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
