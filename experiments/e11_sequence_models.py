"""
Deney 11 - Bolge gecmisini dizi olarak okuyan derin modeller (GRU, 1D-CNN).

Modeller (src/sequence.py):
  GRU_SEQ : bugunku A seti + bolge gecmisi dizisi, GRU ile okunur
  CNN_SEQ : ayni girdi, 1 boyutlu evrisimli ag (1D-CNN) ile okunur
  GRU_HYB : GRU_SEQ + E09'daki el yapimi gecmis ozetleri
Karsilastirma: A + LR (ana model), A+HIST + LR (E09), egitimsiz klor.

Sizinti kurali: yalnizca ayni ZAPE'de en az 2 gun once alinmis numuneler.
Protokoller: ZAPE gruplu arindirilmis CV ve genisleyen pencere (3 tohum).
Ayrica bilinen gecmis numune sayisina gore basari egrisi
(soguk baslangictan sicak baslangica).

Calistirma: python -m experiments.e11_sequence_models
Ciktilar: results/e11_sequence_models/
"""

import numpy as np
import pandas as pd

from src import sequence
from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.history import HISTORY_FEATURES, add_history_features
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e11_sequence_models"
SEEDS = (0, 1, 2)
A = FEATURE_SETS["A"]
RUNS = [("A + LR", A, "LR"),
        ("A + gecmis ozeti + LR", A + HISTORY_FEATURES, "LR"),
        ("GRU (dizi)", A + ["_row"], "GRU_SEQ"),
        ("1D-CNN (dizi)", A + ["_row"], "CNN_SEQ"),
        ("GRU (dizi + ozet)", A + HISTORY_FEATURES + ["_row"], "GRU_HYB")]
BINS = [-1, 0, 2, 5, 10, 10_000]
BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11+"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())
    df["_row"] = df.index
    sequence.STORE = sequence.build_store(df)
    sequence.STORE_HYB = sequence.build_store(df, A + HISTORY_FEATURES)
    df["hist_bin"] = pd.cut(df["HIST_N"], BINS, labels=BIN_LABELS)
    routine = ~df["is_investigation"].values
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine}
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values
    y = df["y"].values

    rolling = list(temporal_splits(df, range(2015, 2026), unseen_zape=False))
    rows, probs = [], {}
    for protocol in ("zape_cv", "rolling"):
        test_rows = None
        for name, cols, model in RUNS:
            if protocol == "zape_cv":
                oof = run_cv(df, cols, model, SEEDS)
            else:
                oof = run_temporal(df, cols, model, rolling, SEEDS)
            probs[(protocol, name)] = mean_prob(oof)
            test_rows = np.unique(oof["row"].values)
            line = []
            for sname, smask in subsets.items():
                m = mean_std(summarize(df, oof, smask))
                rows.append({"protocol": protocol, "model": name,
                             "test": sname, **m})
                line.append(f"{sname}: ROC={m['ROC_AUC']:.3f} "
                            f"PR={m['PR_AUC']:.3f}")
            print(f"{protocol:8s} {name:24s} " + " | ".join(line))
        for sname, smask in subsets.items():
            idx = test_rows[smask[test_rows]]
            m = compute_metrics(y[idx], chlorine[idx])
            m.pop("brier", None)
            rows.append({"protocol": protocol, "model": "Egitimsiz klor",
                         "test": sname, **m})
    pd.DataFrame(rows).round(4).to_csv(OUT / "1_summary.csv", index=False)

    # Soguk -> sicak baslangic egrisi (ZAPE CV)
    curve = []
    for b in BIN_LABELS:
        idx = np.where((df["hist_bin"] == b).values)[0]
        if y[idx].sum() in (0, len(idx)):
            continue
        row = {"bilinen_gecmis": b, "n": len(idx), "n_pos": int(y[idx].sum()),
               "Egitimsiz klor": compute_metrics(y[idx],
                                                 chlorine[idx])["ROC_AUC"]}
        for name, _, _ in RUNS:
            p = probs[("zape_cv", name)].reindex(idx).values
            row[name] = compute_metrics(y[idx], p)["ROC_AUC"]
        curve.append(row)
    curve = pd.DataFrame(curve)
    curve.round(4).to_csv(OUT / "2_ogrenme_egrisi_ROC.csv", index=False)
    print(curve.round(3).to_string(index=False))

    # Eslestirilmis ZAPE bootstrap (ZAPE CV)
    tests = []
    for name in ("GRU (dizi)", "1D-CNN (dizi)", "GRU (dizi + ozet)"):
        p = probs[("zape_cv", name)].sort_index().values
        for ref_name, ref in (
                ("Egitimsiz klor", chlorine),
                ("A + LR", probs[("zape_cv", "A + LR")].sort_index().values),
                ("A + gecmis ozeti + LR",
                 probs[("zape_cv", "A + gecmis ozeti + LR")]
                 .sort_index().values)):
            for sname, smask in subsets.items():
                idx = np.where(smask)[0]
                for metric in ("ROC_AUC", "PR_AUC"):
                    b = zape_bootstrap(df, idx, p[idx], ref[idx],
                                       metric=metric)
                    tests.append({"karsilastirma": f"{name} - {ref_name}",
                                  "test": sname, "metrik": metric, **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "3_bootstrap.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
