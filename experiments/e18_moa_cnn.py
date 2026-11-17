"""
Deney 18 - Yayimlanmis (orijinal) tablo 1D-CNN'i ile yeniden deneme.

Danisman, E12'deki kendi yazdigimiz 1D-CNN'in dusuk sonucunu sordu. Bu deneyde
Kaggle MoA yarismasi 2. sira cozumundeki "soft-ordering" 1D-CNN orijinal
mimarisi ve egitim ayarlariyla (src/moa_cnn.py) ayni protokolde calistirilir.

Protokol: ZAPE gruplu, arindirilmis 5 katli CV, 3 tohum (E12/E13 ile ayni
bolmeler). Ozellik setleri: A (klor + hava + ay) ve A+HIST (+ bolge gecmisi).
ROC-AUC ve PR-AUC esige bagli olmadigi icin ic dogrulamayla esik secimi
yapilmaz (tune_threshold=False); bu sure tasarrufu icindir.

Calistirma: python -m experiments.e18_moa_cnn A      (veya A+HIST)
Ciktilar: results/e18_moa_cnn/
"""

import sys

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import mean_prob, mean_std, run_cv, summarize, zape_bootstrap
from src.history import HISTORY_FEATURES, add_history_features

OUT = RESULTS_DIR / "e18_moa_cnn"
SEEDS = (0, 1, 2)
SETS = {"A": FEATURE_SETS["A"], "A+HIST": FEATURE_SETS["A"] + HISTORY_FEATURES}


def main(fs):
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())
    subsets = {"Tumu": np.ones(len(df), bool),
               "Rutin": ~df["is_investigation"].values}
    oof = run_cv(df, SETS[fs], "MOA_CNN", SEEDS, tune_threshold=False)
    tag = fs.replace("+", "_")
    oof.to_csv(OUT / f"oof_{tag}.csv", index=False)
    rows = []
    for sname, smask in subsets.items():
        m = mean_std(summarize(df, oof, smask))
        rows.append({"model": "MOA_CNN", "features": fs, "test": sname,
                     "ROC_AUC": m["ROC_AUC"], "PR_AUC": m["PR_AUC"]})
    pd.DataFrame(rows).round(4).to_csv(OUT / f"summary_{tag}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values
    p = mean_prob(oof).sort_index().values
    tests = []
    for sname, smask in subsets.items():
        idx = np.where(smask)[0]
        for metric in ("ROC_AUC", "PR_AUC"):
            b = zape_bootstrap(df, idx, p[idx], chlorine[idx], metric=metric)
            tests.append({"karsilastirma": f"MOA_CNN {fs} - YALNIZ_KLOR",
                          "test": sname, "metrik": metric, **b})
    pd.DataFrame(tests).round(4).to_csv(OUT / f"bootstrap_{tag}.csv",
                                        index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
