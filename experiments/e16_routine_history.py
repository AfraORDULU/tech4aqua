"""
Deney 16 - Bolge gecmisi yalnizca rutin numunelerden hesaplanirsa.

E15 hata analizi, bolge gecmisinden gelen kazancin bir kisminin
sorusturma (inceleme) baglamindan gelebilecegini gosterdi: sorusturma
numuneleri zaten bir sorun nedeniyle alinir ve gecmis ozetine girdiklerinde
"bu bolgede sorun var" bilgisini tasirlar. Burada gecmis ozetleri yalnizca
ONCEKI RUTIN numunelerden hesaplanir (HISTR_*). Kazanc surerse, bolge
gecmisinin katkisi sorusturma baglamindan bagimsizdir.

Modeller: LR ve TabPFN; ozellik setleri A+HIST (tum gecmis) ve
A+HISTR (yalniz rutin gecmis). ZAPE gruplu arindirilmis CV, 3 tohum.

Calistirma: TABPFN_MODEL_PATH=/yol/tabpfn-v2-classifier.ckpt \
            python -m experiments.e16_routine_history
Ciktilar: results/e16_routine_history/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            summarize, zape_bootstrap)
from src.history import HISTORY_FEATURES, add_history_features

OUT = RESULTS_DIR / "e16_routine_history"
SEEDS = (0, 1, 2)
A = FEATURE_SETS["A"]
HISTR = [c.replace("HIST_", "HISTR_") for c in HISTORY_FEATURES]
SETS = {"A": A, "A+HIST": A + HISTORY_FEATURES, "A+HISTR": A + HISTR}
RUNS = [("LR", "A+HISTR"), ("TABPFN", "A+HISTR")]
E13_OOF = RESULTS_DIR / "e13_tabpfn_history" / "oof.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_dataset()
    df = add_history_features(base)
    df = add_history_features(df, source_mask=~df["is_investigation"].values,
                              prefix="HISTR_")
    routine = ~df["is_investigation"].values
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine,
               "Inceleme": ~routine}
    cl = df[CHLORINE]
    chlorine = -cl.fillna(cl.median()).values
    y = df["y"].values

    rows, probs, oofs = [], {}, []
    e13 = pd.read_csv(E13_OOF)
    e13 = e13[e13["protocol"] == "zape_cv"]
    for model, fs in [("LR", "A"), ("LR", "A+HIST"), ("TABPFN", "A"),
                      ("TABPFN", "A+HIST")]:
        g = e13[(e13["model"] == model) & (e13["feature_set"] == fs)]
        probs[(model, fs)] = mean_prob(g).sort_index().values
        for sname, smask in subsets.items():
            m = mean_std(summarize(df, g, smask))
            rows.append({"model": model, "feature_set": fs, "test": sname,
                         "kaynak": "E13", **m})
    for model, fs in RUNS:
        oof = run_cv(df, SETS[fs], model, SEEDS)
        oofs.append(oof.assign(model=model, feature_set=fs))
        probs[(model, fs)] = mean_prob(oof).sort_index().values
        line = []
        for sname, smask in subsets.items():
            m = mean_std(summarize(df, oof, smask))
            rows.append({"model": model, "feature_set": fs, "test": sname,
                         "kaynak": "E16", **m})
            line.append(f"{sname}: ROC={m['ROC_AUC']:.3f} "
                        f"PR={m['PR_AUC']:.3f} MCC={m['MCC']:.3f}")
        print(f"{model:6s} {fs:8s} " + " | ".join(line))
    for sname, smask in subsets.items():
        m = compute_metrics(y[smask], chlorine[smask])
        m.pop("brier", None)
        rows.append({"model": "YALNIZ_KLOR", "feature_set": "-",
                     "test": sname, "kaynak": "-", **m})
    pd.DataFrame(rows).round(4).to_csv(OUT / "summary.csv", index=False)
    pd.concat(oofs).to_csv(OUT / "oof.csv", index=False)

    # Gecmisi olan numune orani: rutin gecmis daha seyrektir
    pd.DataFrame({
        "gecmis": ["Tum gecmis", "Yalniz rutin gecmis"],
        "gecmisi_olan_numune": [int((df["HIST_N"] > 0).sum()),
                                int((df["HISTR_N"] > 0).sum())],
        "gecmisi_olan_rutin": [int((df["HIST_N"][routine] > 0).sum()),
                               int((df["HISTR_N"][routine] > 0).sum())],
        "gecmisi_olan_inceleme": [int((df["HIST_N"][~routine] > 0).sum()),
                                  int((df["HISTR_N"][~routine] > 0).sum())],
    }).to_csv(OUT / "gecmis_kapsami.csv", index=False)

    tests = []
    for a, b in ((("LR", "A+HISTR"), ("LR", "A")),
                 (("LR", "A+HISTR"), ("LR", "A+HIST")),
                 (("TABPFN", "A+HISTR"), ("TABPFN", "A")),
                 (("TABPFN", "A+HISTR"), ("TABPFN", "A+HIST")),
                 (("TABPFN", "A+HISTR"), "YALNIZ_KLOR"),
                 (("LR", "A+HISTR"), "YALNIZ_KLOR")):
        ref = chlorine if b == "YALNIZ_KLOR" else probs[b]
        bname = b if b == "YALNIZ_KLOR" else " ".join(b)
        for sname, smask in subsets.items():
            idx = np.where(smask)[0]
            for metric in ("ROC_AUC", "PR_AUC"):
                r = zape_bootstrap(df, idx, probs[a][idx], ref[idx],
                                   metric=metric)
                tests.append({"karsilastirma": f"{' '.join(a)} - {bname}",
                              "test": sname, "metrik": metric, **r})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "bootstrap.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
