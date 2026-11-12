"""
Deney 14 - Dizi modellerinin gelistirilmesi: ayarlarin egitim verisi
icinde secilmesi.

E11'de dizi modelleri tek bir sabit ayarla (16 adim, 200 epoch) denendi.
Burada her egitim katinda dizi uzunlugu (4, 8, 16) ve epoch sayisi (100,
200) ZAPE gruplu 3 katli ic dogrulamayla, PR-AUC'ye gore secilir
(src/sequence.SelectedSequenceClassifier). Test verisi secimde kullanilmaz.

Modeller: GRU, 1D-CNN (dizi) ve hibrit GRU (dizi + el yapimi gecmis ozeti).
Hesaplama maliyeti nedeniyle esik 0.5 sabit tutulur (ROC/PR-AUC esikten
bagimsizdir). GRU modelleri ayrica genisleyen pencereyle (gelecek yil)
test edilir. Test disi olasiliklar oof.csv'ye yazilir.

Calistirma: python -m experiments.e14_sequence_selection
Ciktilar: results/e14_sequence_selection/
"""

from collections import Counter

import numpy as np
import pandas as pd

from src import sequence
from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.history import HISTORY_FEATURES, add_history_features
from src.models import make_model
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e14_sequence_selection"
SEEDS = (0, 1, 2)
A = FEATURE_SETS["A"]
RUNS = [("GRU (secimli)", A + ["_row"], "GRU_SEL"),
        ("1D-CNN (secimli)", A + ["_row"], "CNN_SEL"),
        ("GRU dizi + ozet (secimli)", A + HISTORY_FEATURES + ["_row"],
         "GRU_HYB_SEL")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())
    df["_row"] = df.index
    sequence.STORE = sequence.build_store(df)
    sequence.STORE_HYB = sequence.build_store(df, A + HISTORY_FEATURES)
    routine = ~df["is_investigation"].values
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine}
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values

    rows, probs, oofs = [], {}, []
    for name, cols, model in RUNS:
        oof = run_cv(df, cols, model, SEEDS, tune_threshold=False)
        oofs.append(oof.assign(protocol="zape_cv", model=name))
        probs[name] = mean_prob(oof).sort_index().values
        line = []
        for sname, smask in subsets.items():
            m = mean_std(summarize(df, oof, smask))
            rows.append({"model": name, "test": sname, **m})
            line.append(f"{sname}: ROC={m['ROC_AUC']:.3f} "
                        f"PR={m['PR_AUC']:.3f}")
        print(f"{name:28s} " + " | ".join(line))
    pd.DataFrame(rows).round(4).to_csv(OUT / "summary.csv", index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)

    # Gelecek yil testi (genisleyen pencere), GRU modelleri
    rolling = list(temporal_splits(df, range(2015, 2026), unseen_zape=False))
    roll_rows = []
    for name, cols, model in RUNS:
        if not model.startswith("GRU"):
            continue
        oof = run_temporal(df, cols, model, rolling, SEEDS,
                           tune_threshold=False)
        oofs.append(oof.assign(protocol="rolling", model=name))
        test_rows = np.unique(oof["row"].values)
        for sname, smask in subsets.items():
            m = mean_std(summarize(df, oof, smask))
            roll_rows.append({"model": name, "test": sname, **m})
            print(f"rolling {name:28s} {sname}: ROC={m['ROC_AUC']:.3f} "
                  f"PR={m['PR_AUC']:.3f}")
    for sname, smask in subsets.items():
        idx = test_rows[smask[test_rows]]
        m = compute_metrics(df["y"].values[idx], chlorine[idx])
        m.pop("brier", None)
        roll_rows.append({"model": "YALNIZ_KLOR", "test": sname, **m})
    pd.DataFrame(roll_rows).round(4).to_csv(OUT / "summary_rolling.csv",
                                            index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)

    # Tum veriyle egitimde hangi ayar seciliyor? (bilgi amacli)
    chosen = Counter()
    for name, cols, model in RUNS:
        for seed in SEEDS:
            m = make_model(model, seed)
            m.fit(df[cols], df["y"].values)
            chosen[(name, m.best_)] += 1
    pd.DataFrame([{"model": k[0], "dizi_uzunlugu": k[1][0],
                   "epoch": k[1][1], "secilme": v}
                  for k, v in chosen.items()]).to_csv(
        OUT / "secilen_ayarlar_tum_veri.csv", index=False)

    tests = []
    for name in probs:
        for sname, smask in subsets.items():
            idx = np.where(smask)[0]
            for metric in ("ROC_AUC", "PR_AUC"):
                b = zape_bootstrap(df, idx, probs[name][idx], chlorine[idx],
                                   metric=metric)
                tests.append({"karsilastirma": f"{name} - YALNIZ_KLOR",
                              "test": sname, "metrik": metric, **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "bootstrap_vs_klor.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
