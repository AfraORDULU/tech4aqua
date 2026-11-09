"""
Deney 10 - Kesirli koliform sonuclari icin duyarlilik testi
(ara rapordan kalan acik madde).

Veride koliform sonucu 0.2-0.7 arasinda olan 5 numune var
(results/data_audit/03b_fractional_results.csv). Besi de sorusturma
numunesi ve ana analizde pozitif (> 0) sayiliyor. Bu numunelerin
olcum belirsizligi nedeniyle uc senaryo karsilastirilir:
  pozitif   : ana analiz (> 0 ise pozitif)
  negatif   : kesirli sonuclar negatif sayilir (< 1 CFU/100 mL)
  cikarilmis: kesirli sonuclu numuneler veriden cikarilir

Calistirma: python -m experiments.e10_fractional_labels
Ciktilar: results/e10_fractional_labels/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, TARGET_RAW, load_dataset
from src.evaluation import compute_metrics, mean_std, run_cv, summarize

OUT = RESULTS_DIR / "e10_fractional_labels"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_dataset()
    frac = (base[TARGET_RAW] > 0) & (base[TARGET_RAW] < 1)
    print(f"Kesirli sonuclu numune: {int(frac.sum())}")
    base.loc[frac, ["zape", "date", TARGET_RAW, "is_investigation"]].to_csv(
        OUT / "kesirli_numuneler.csv", index=False)

    scenarios = {
        "pozitif": base,
        "negatif": base.assign(y=np.where(frac, 0, base["y"])),
        "cikarilmis": base[~frac].reset_index(drop=True),
    }
    rows = []
    for name, df in scenarios.items():
        routine = ~df["is_investigation"].values
        cl = df[CHLORINE]
        chlorine = -cl.fillna(cl.median()).values
        for model in ("LR", "LGBM"):
            oof = run_cv(df, FEATURE_SETS["A"], model, (0, 1, 2, 3, 4))
            m = mean_std(summarize(df, oof))
            rows.append({"senaryo": name, "model": model,
                         "n_pos": int(df["y"].sum()), **m})
            print(f"{name:10s} {model:4s} ROC={m['ROC_AUC']:.3f} "
                  f"PR={m['PR_AUC']:.3f} MCC={m['MCC']:.3f}")
        c = compute_metrics(df["y"].values, chlorine)
        rows.append({"senaryo": name, "model": "YALNIZ_KLOR",
                     "n_pos": int(df["y"].sum()),
                     "ROC_AUC": c["ROC_AUC"], "PR_AUC": c["PR_AUC"]})
        _ = routine  # kesirli numunelerin hepsi sorusturma; rutin degismez
    pd.DataFrame(rows).round(4).to_csv(OUT / "summary.csv", index=False)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
