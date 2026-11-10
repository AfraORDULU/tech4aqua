"""
Deney 13 - TabPFN ile bolge gecmisi.

TabPFN (E12) yeni bolge testinde en iyi temel model olmustu; bolge gecmisi
(E09) ise klorun onune gecen ilk bilgiydi. Burada ikisi birlikte denenir.

Modeller: LR ve TabPFN; ozellik setleri: A, A+HIST, KLOR+HIST.
Protokoller: ZAPE gruplu arindirilmis CV ve genisleyen pencere (3 tohum).
Test disi olasiliklar oof.csv'ye yazilir (E15 hata analizi ve kalibrasyon
bunu kullanir).

Calistirma: TABPFN_MODEL_PATH=/yol/tabpfn-v2-classifier.ckpt \
            python -m experiments.e13_tabpfn_history
Ciktilar: results/e13_tabpfn_history/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.history import HISTORY_FEATURES, add_history_features
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e13_tabpfn_history"
SEEDS = (0, 1, 2)
SETS = {"A": FEATURE_SETS["A"],
        "A+HIST": FEATURE_SETS["A"] + HISTORY_FEATURES,
        "KLOR+HIST": [CHLORINE, "HIST_POS_RATE", "HIST_DAYS_LAST_POS"]}
RUNS = [("LR", "A"), ("LR", "A+HIST"), ("TABPFN", "A"),
        ("TABPFN", "A+HIST"), ("TABPFN", "KLOR+HIST")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())
    routine = ~df["is_investigation"].values
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine}
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values
    y = df["y"].values
    rolling = list(temporal_splits(df, range(2015, 2026), unseen_zape=False))

    rows, probs, oofs = [], {}, []
    for protocol in ("zape_cv", "rolling"):
        test_rows = None
        for model, fs in RUNS:
            oof = (run_cv(df, SETS[fs], model, SEEDS) if protocol == "zape_cv"
                   else run_temporal(df, SETS[fs], model, rolling, SEEDS))
            probs[(protocol, model, fs)] = mean_prob(oof)
            oofs.append(oof.assign(protocol=protocol, model=model,
                                   feature_set=fs))
            test_rows = np.unique(oof["row"].values)
            line = []
            for sname, smask in subsets.items():
                m = mean_std(summarize(df, oof, smask))
                rows.append({"protocol": protocol, "model": model,
                             "feature_set": fs, "test": sname, **m})
                line.append(f"{sname}: ROC={m['ROC_AUC']:.3f} "
                            f"PR={m['PR_AUC']:.3f} MCC={m['MCC']:.3f}")
            print(f"{protocol:8s} {model:6s} {fs:9s} " + " | ".join(line))
        for sname, smask in subsets.items():
            idx = test_rows[smask[test_rows]]
            m = compute_metrics(y[idx], chlorine[idx])
            m.pop("brier", None)
            rows.append({"protocol": protocol, "model": "YALNIZ_KLOR",
                         "feature_set": "-", "test": sname, **m})
    pd.DataFrame(rows).round(4).to_csv(OUT / "summary.csv", index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)

    tests = []
    for (model, fs), (rmodel, rfs) in (
            (("TABPFN", "A+HIST"), ("LR", "A+HIST")),
            (("TABPFN", "A+HIST"), ("TABPFN", "A")),
            (("TABPFN", "KLOR+HIST"), ("LR", "A+HIST"))):
        p = probs[("zape_cv", model, fs)].sort_index().values
        r = probs[("zape_cv", rmodel, rfs)].sort_index().values
        for ref_name, ref in ((f"{rmodel} {rfs}", r),
                              ("YALNIZ_KLOR", chlorine)):
            for sname, smask in subsets.items():
                idx = np.where(smask)[0]
                for metric in ("ROC_AUC", "PR_AUC"):
                    b = zape_bootstrap(df, idx, p[idx], ref[idx],
                                       metric=metric)
                    tests.append({"karsilastirma":
                                  f"{model} {fs} - {ref_name}",
                                  "test": sname, "metrik": metric, **b})
    t = pd.DataFrame(tests).drop_duplicates(
        subset=["karsilastirma", "test", "metrik"])
    t.round(4).to_csv(OUT / "bootstrap.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
