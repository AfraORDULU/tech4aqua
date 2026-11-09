"""
Deney 7b - E07 tamamlayicisi: M3'un egitimsiz klora karsi testi ve
M1'in neden egitimsiz klordan dusuk ciktigi.

1) M3 (klor + sicaklik + termal) ile egitimsiz klor siralamasi ayni
   numunelerde, ZAPE duzeyinde eslestirilmis bootstrap ile karsilastirilir.
2) M1 (yalniz klorla egitilmis LR) her katin icinde numuneleri klorla
   birebir ayni sirada dizer (tek degiskenli, monoton model). Birlesik
   (pooled) sonucunun dusuk cikmasi, 5 farkli kat modelinin olasilik
   olceklerinin farkli olmasindan kaynaklanir. Bunu gostermek icin kat
   ici sira korelasyonu ve kat bazinda ortalama AUC hesaplanir.

On kosul: python -m experiments.e07_chlorine_temperature
Calistirma: python -m experiments.e07b_m3_vs_chlorine
Ciktilar: results/e07_chlorine_temperature/ (7b_ on ekli dosyalar)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import RESULTS_DIR
from src.data import CHLORINE, load_dataset
from src.evaluation import zape_bootstrap

DIR = RESULTS_DIR / "e07_chlorine_temperature"


def main():
    df = load_dataset()
    y = df["y"].values
    routine = ~df["is_investigation"].values
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values

    m1 = pd.read_csv(DIR / "oof_M1_klor.csv")
    m3 = pd.read_csv(DIR / "oof_M3_klor_sicaklik_termal.csv")
    m3_mean = m3.groupby("row")["prob"].mean().sort_index().values
    m1_mean = m1.groupby("row")["prob"].mean().sort_index().values

    # 1) M3 - egitimsiz klor
    tests = []
    for sname, smask in (("Tumu", np.ones(len(df), bool)),
                         ("Rutin", routine)):
        idx = np.where(smask)[0]
        for metric in ("ROC_AUC", "PR_AUC"):
            for name, ref in (("M3 - YALNIZ_KLOR", chlorine),
                              ("M1 - YALNIZ_KLOR", None)):
                if ref is None:
                    b = zape_bootstrap(df, idx, m1_mean[idx], chlorine[idx],
                                       metric=metric)
                else:
                    b = zape_bootstrap(df, idx, m3_mean[idx], ref[idx],
                                       metric=metric)
                tests.append({"karsilastirma": name, "test": sname,
                              "metrik": metric,
                              "zape_sayisi": df["zape"].values[idx].size and
                              len(set(df["zape"].values[idx])), **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(DIR / "7b_bootstrap_vs_egitimsiz_klor.csv",
                      index=False)
    print(t.round(3).to_string(index=False))

    # 2) M1 havuzlama tanisi
    diag = []
    for seed, g in m1.groupby("seed"):
        for fold, h in g.groupby("fold"):
            rho = spearmanr(h["prob"], chlorine[h["row"].values]).correlation
            diag.append({"seed": seed, "fold": fold, "spearman_M1_klor": rho})
    diag = pd.DataFrame(diag)
    pooled_r, pooled_c = [], []
    for seed, g in m1.groupby("seed"):
        g = g[routine[g["row"].values]]
        pooled_r.append(roc_auc_score(y[g["row"]], g["prob"]))
        pooled_c.append(average_precision_score(y[g["row"]], g["prob"]))
    summary = pd.DataFrame([{
        "kat_ici_spearman_min": diag["spearman_M1_klor"].min(),
        "kat_ici_spearman_ortalama": diag["spearman_M1_klor"].mean(),
        "M1_rutin_ROC_birlesik": np.mean(pooled_r),
        "M1_rutin_AP_birlesik": np.mean(pooled_c),
        "klor_rutin_ROC": roc_auc_score(y[routine], chlorine[routine]),
        "klor_rutin_AP": average_precision_score(y[routine],
                                                 chlorine[routine]),
    }])
    summary.round(4).to_csv(DIR / "7b_M1_havuzlama_tanisi.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
