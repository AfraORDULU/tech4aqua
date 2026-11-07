"""
Deney 5 - Model secimi: klasik modeller, derin ogrenme ve alan genelleme.

Modeller : LR, LGBM (klasik), MLP (derin, ERM), GroupDRO, CORAL
           (alan genelleme; alan = belediye). Ayrica egitimsiz
           YALNIZ_KLOR siralamasi referans olarak verilir.
Setler   : A (klor + hava + ay), C (A + alan bilgisi bayraklari)
Protokoller
  zape_cv        : ZAPE gruplu, arindirilmis 5 katli CV (5 tohum)
  rolling_unseen : gecmisle egit, sonraki yili test et; test yilinin
                   ZAPE'leri egitimde yok (3 tohum)
  lomo           : leave-one-municipality-out; iki belediyeyle egit,
                   ucuncusunda test et (3 tohum). En zor alan kaymasi.

Calistirma: python -m experiments.e05_model_comparison
Ciktilar: results/e05_model_comparison/
"""

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            run_temporal, summarize, zape_bootstrap)
from src.validation import temporal_splits

OUT = RESULTS_DIR / "e05_model_comparison"
MODELS = ("LR", "LGBM", "MLP", "GROUPDRO", "CORAL")


def lomo_splits(df):
    towns = df["municipality"].values
    for town in np.unique(towns):
        yield town, np.where(towns != town)[0], np.where(towns == town)[0]


def chlorine_metrics(df, rows, subset):
    idx = rows[subset[rows]]
    cl = df[CHLORINE]
    m = compute_metrics(df["y"].values[idx],
                        -cl.fillna(cl.median()).values[idx])
    m.pop("brier", None)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    routine = ~df["is_investigation"].values
    no2025 = df["year"].values < 2025
    subsets = {"Tumu": np.ones(len(df), bool), "Rutin": routine,
               "Tumu_2025_haric": no2025}

    protocols = {
        "zape_cv": None,
        "rolling_unseen": list(temporal_splits(df, range(2015, 2026),
                                               unseen_zape=True)),
        "lomo": list(lomo_splits(df)),
    }

    rows, probs = [], {}
    for protocol, splits in protocols.items():
        for fs in ("A", "C"):
            for model in MODELS:
                if splits is None:
                    oof = run_cv(df, FEATURE_SETS[fs], model,
                                 (0, 1, 2, 3, 4))
                else:
                    oof = run_temporal(df, FEATURE_SETS[fs], model, splits,
                                       (0, 1, 2))
                probs[(protocol, fs, model)] = mean_prob(oof)
                test_rows = np.unique(oof["row"].values)
                for sname, smask in subsets.items():
                    if df["y"].values[test_rows[smask[test_rows]]].sum() < 3:
                        continue
                    m = mean_std(summarize(df, oof, smask))
                    rows.append({"protocol": protocol, "feature_set": fs,
                                 "model": model, "test": sname, **m})
                    if model == "LR" and fs == "A":
                        rows.append({"protocol": protocol, "feature_set": "-",
                                     "model": "YALNIZ_KLOR", "test": sname,
                                     **chlorine_metrics(df, test_rows,
                                                        smask)})
                r = [x for x in rows if x["model"] == model
                     and x["protocol"] == protocol
                     and x["feature_set"] == fs]
                msg = " | ".join(f"{x['test']}: ROC={x['ROC_AUC']:.3f} "
                                 f"PR={x['PR_AUC']:.3f}" for x in r)
                print(f"{protocol:14s} {fs} {model:8s} {msg}")

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(OUT / "summary.csv", index=False)

    # LOMO: belediye bazinda ayrintili sonuc (A seti)
    lomo = []
    for town, _, te in protocols["lomo"]:
        y = df["y"].values[te]
        for model in MODELS:
            p = probs[("lomo", "A", model)].reindex(te).values
            lomo.append({"test_belediye": town, "model": model,
                         **compute_metrics(y, p)})
        cl = df[CHLORINE]
        lomo.append({"test_belediye": town, "model": "YALNIZ_KLOR",
                     **compute_metrics(y, -cl.fillna(cl.median())
                                       .values[te])})
    pd.DataFrame(lomo).round(4).to_csv(OUT / "lomo_by_town.csv",
                                       index=False)

    # Eslestirilmis bootstrap: her model - LR (zape_cv, A seti)
    tests = []
    idx = np.arange(len(df))
    lr = probs[("zape_cv", "A", "LR")].sort_index().values
    for model in MODELS[1:]:
        p = probs[("zape_cv", "A", model)].sort_index().values
        for sname, smask in (("Tumu", subsets["Tumu"]), ("Rutin", routine)):
            for metric in ("ROC_AUC", "PR_AUC"):
                b = zape_bootstrap(df, idx[smask], p[smask], lr[smask],
                                   metric=metric)
                tests.append({"karsilastirma": f"{model} - LR",
                              "test": sname, "metrik": metric, **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "bootstrap_vs_LR.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
