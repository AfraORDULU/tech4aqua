"""
Deney 1 - Ara rapordaki temel sonuclarin yeniden uretilmesi.

3 model (LR, RF, LGBM) x 3 ozellik seti (A, B, C), 5 tohum,
ZAPE gruplu 5 katli CV. Arindirma (ayni belediye-gun) acik ve kapali
iki halde calistirilir; farki sizintinin etkisini gosterir.

Calistirma (proje kok klasorunden):
    python -m experiments.e01_baseline

Ciktilar: results/e01_baseline/
    summary.csv  : her kombinasyon icin ortalama +- std
    oof.csv      : her (seed, numune) icin test-disi olasilik
"""

import pandas as pd

from src.config import RESULTS_DIR
from src.data import FEATURE_SETS, load_dataset
from src.evaluation import mean_std, run_cv, summarize
from src.models import MODEL_NAMES

OUT = RESULTS_DIR / "e01_baseline"
SEEDS = (0, 1, 2, 3, 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"Veri: {len(df)} numune, {df['y'].sum()} pozitif, "
          f"{df['zape'].nunique()} ZAPE")

    summaries, oofs = [], []
    for purge in (True, False):
        for fs in ("A", "B", "C"):
            for model in MODEL_NAMES:
                oof = run_cv(df, FEATURE_SETS[fs], model, SEEDS,
                             purge=purge)
                row = {"protocol": "purged" if purge else "zape_only",
                       "feature_set": fs, "model": model}
                row.update(mean_std(summarize(df, oof)))
                summaries.append(row)
                oofs.append(oof.assign(**{k: row[k] for k in
                                          ("protocol", "feature_set",
                                           "model")}))
                print(f"{row['protocol']:9s} {fs} {model:4s} "
                      f"ROC={row['ROC_AUC']:.3f} PR={row['PR_AUC']:.3f} "
                      f"MCC={row['MCC']:.3f} MCC@0.5={row['MCC_05']:.3f}")

    pd.DataFrame(summaries).round(4).to_csv(OUT / "summary.csv",
                                            index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
