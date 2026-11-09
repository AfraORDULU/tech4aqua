"""
Deney 9 - Warm-start fizibilitesi: bolge gecmisi model ve klora bir sey
katiyor mu? (E08 numarasi, ZAPE-ay onceliklendirme onerisine ayrildi.)

Ozellikler: src/history.py. Bir numune icin yalnizca ayni ZAPE'de,
en az 2 gun once alinmis (sonucu karar aninda bilinen) numuneler kullanilir.

Protokoller
  zape_cv : ZAPE gruplu, arindirilmis 5 katli CV (5 tohum). Model
            agirliklari baska ZAPE'lerden ogrenilir; test ZAPE'sinin
            gecmisi yalnizca girdi olarak kullanilir (yeniden egitim yok).
  rolling : Gecmis yillarla egit, sonraki yili test et (3 tohum).
            Bolge gecmisi bu senaryonun dogal halidir; bu yuzden
            gorulmemis ZAPE kurali uygulanmaz.

Ozellik setleri
  A            : klor + hava + ay (ana model)
  A+HIST       : A + bolge gecmisi
  KLOR+HIST    : klor + gecmis pozitif orani + son pozitiften gecen gun
Referanslar (egitimsiz): YALNIZ_KLOR (-klor), YALNIZ_GECMIS (gecmis pozitif orani)

Ciktilar ayrica bilinen gecmis numune sayisina (HIST_N) gore kirilir:
soguk baslangictan sicak baslangica egri.

Calistirma: python -m experiments.e09_warm_start
Ciktilar: results/e09_warm_start/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, check_features, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.history import HISTORY_FEATURES, add_history_features
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e09_warm_start"
SETS = {
    "A": FEATURE_SETS["A"],
    "A+HIST": FEATURE_SETS["A"] + HISTORY_FEATURES,
    "KLOR+HIST": [CHLORINE, "HIST_POS_RATE", "HIST_DAYS_LAST_POS"],
}
RUNS = [("A", "LR"), ("A+HIST", "LR"), ("A+HIST", "LGBM"),
        ("A+HIST", "MLP"), ("KLOR+HIST", "LR")]
BINS = [-1, 0, 2, 5, 10, 10_000]
BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11+"]


def reference_scores(df):
    cl = df[CHLORINE]
    return {
        "YALNIZ_KLOR": -cl.fillna(cl.median()).values,
        # gecmisi olmayan numune icin onsel oran (0.1)
        "YALNIZ_GECMIS": df["HIST_POS_RATE"].fillna(0.1).values,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset())
    for cols in SETS.values():
        check_features(cols)
    df["hist_bin"] = pd.cut(df["HIST_N"], BINS, labels=BIN_LABELS)
    routine = ~df["is_investigation"].values
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine,
               "Sorusturma": ~routine}

    # Gecmis miktarina gore numune dagilimi
    dist = pd.crosstab([df["is_investigation"].map(
        {False: "Rutin", True: "Sorusturma"}), df["hist_bin"]], df["y"])
    dist.columns = ["negatif", "pozitif"]
    dist.to_csv(OUT / "0_gecmis_dagilimi.csv")

    refs = reference_scores(df)
    rolling = list(temporal_splits(df, range(2015, 2026), unseen_zape=False))
    rows, probs = [], {}
    for protocol in ("zape_cv", "rolling"):
        test_rows = None
        for fs, model in RUNS:
            if protocol == "zape_cv":
                oof = run_cv(df, SETS[fs], model, (0, 1, 2, 3, 4))
            else:
                oof = run_temporal(df, SETS[fs], model, rolling, (0, 1, 2))
            probs[(protocol, fs, model)] = mean_prob(oof)
            test_rows = np.unique(oof["row"].values)
            for sname, smask in subsets.items():
                m = mean_std(summarize(df, oof, smask))
                rows.append({"protocol": protocol, "feature_set": fs,
                             "model": model, "test": sname, **m})
            r = [x for x in rows[-3:]]
            print(f"{protocol:8s} {fs:10s} {model:5s} " + " | ".join(
                f"{x['test']}: ROC={x['ROC_AUC']:.3f} PR={x['PR_AUC']:.3f}"
                for x in r))
        for name, score in refs.items():
            for sname, smask in subsets.items():
                idx = test_rows[smask[test_rows]]
                m = compute_metrics(df["y"].values[idx], score[idx])
                m.pop("brier", None)
                rows.append({"protocol": protocol, "feature_set": "-",
                             "model": name, "test": sname, **m})
    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(OUT / "1_summary.csv", index=False)

    # Soguk -> sicak baslangic egrisi (zape_cv, tohum ortalamasi olasilik)
    curve = []
    y = df["y"].values
    for b in BIN_LABELS:
        for sname, smask in (("Tumu", subsets["Tumu"]), ("Rutin", routine)):
            idx = np.where((df["hist_bin"] == b).values & smask)[0]
            if y[idx].sum() == 0 or y[idx].sum() == len(idx):
                continue
            row = {"gecmis_numune": b, "test": sname, "n": len(idx),
                   "n_pos": int(y[idx].sum())}
            for fs, model in (("A", "LR"), ("A+HIST", "LR"),
                              ("KLOR+HIST", "LR")):
                p = probs[("zape_cv", fs, model)].reindex(idx).values
                row[f"ROC_{fs}_{model}"] = compute_metrics(y[idx], p)["ROC_AUC"]
            row["ROC_YALNIZ_KLOR"] = compute_metrics(
                y[idx], refs["YALNIZ_KLOR"][idx])["ROC_AUC"]
            curve.append(row)
    pd.DataFrame(curve).round(4).to_csv(OUT / "2_soguk_sicak_egrisi.csv",
                                        index=False)

    # Eslestirilmis ZAPE bootstrap (zape_cv)
    tests = []
    base = probs[("zape_cv", "A", "LR")].sort_index().values
    for fs, model in (("A+HIST", "LR"), ("KLOR+HIST", "LR"),
                      ("A+HIST", "LGBM"), ("A+HIST", "MLP")):
        p = probs[("zape_cv", fs, model)].sort_index().values
        for sname, smask in (("Tumu", subsets["Tumu"]), ("Rutin", routine)):
            idx = np.where(smask)[0]
            for ref_name, ref in (("A_LR", base),
                                  ("YALNIZ_KLOR", refs["YALNIZ_KLOR"])):
                for metric in ("ROC_AUC", "PR_AUC"):
                    b = zape_bootstrap(df, idx, p[idx], ref[idx],
                                       metric=metric)
                    tests.append({"karsilastirma": f"{fs}_{model} - {ref_name}",
                                  "test": sname, "metrik": metric, **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "3_bootstrap.csv", index=False)
    print(pd.DataFrame(curve).round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
