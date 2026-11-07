"""
Deney 2 - Rutin numuneler ve yalniz klor kiyasi (hocanin 1. ve 2. maddesi).

Bolum 1 - Ornekleme yapisi
    Pozitiflerin ne kadari sorusturma numunesinden geliyor?
    Ozellikler numunenin sorusturma numunesi olup olmadigini
    tahmin edebiliyor mu? (Edebiliyorsa model "olay sonrasi mi alinmis"
    sorusunu ogreniyor olabilir.)

Bolum 2 - Rutin numunelerde performans
    a) Tum veriyle egitilen modeller (e01) yalniz rutin numunelerde
       degerlendirilir.
    b) Yalniz rutin numunelerle egitilip rutin numunelerde test edilir.

Bolum 3 - Yalniz klor
    Hic egitim yapmadan numuneler klor degerine gore siralanir
    (dusuk klor = yuksek risk). ROC-AUC ve PR-AUC hesaplanir.
    Yasal kural (klor <= 0.2 mg/L ise pozitif) icin MCC hesaplanir.
    Model ile klor arasindaki fark ZAPE duzeyinde eslestirilmis
    bootstrap ile test edilir.

On kosul: python -m experiments.e01_baseline
Calistirma: python -m experiments.e02_routine_and_chlorine
Ciktilar: results/e02_routine_and_chlorine/
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import (compute_metrics, mean_prob, mean_std, run_cv,
                            summarize, zape_bootstrap)
from src.models import MODEL_NAMES

OUT = RESULTS_DIR / "e02_routine_and_chlorine"
E01_OOF = RESULTS_DIR / "e01_baseline" / "oof.csv"
SEEDS = (0, 1, 2, 3, 4)
LEGAL_LIMIT = 0.2  # mg/L


def sampling_structure(df):
    rows = []
    for name, mask in (("Tumu", np.ones(len(df), bool)),
                       ("Rutin", ~df["is_investigation"].values),
                       ("Sorusturma", df["is_investigation"].values)):
        sub = df[mask]
        rows.append({"grup": name, "numune": len(sub),
                     "pozitif": int(sub["y"].sum()),
                     "pozitif_orani": sub["y"].mean(),
                     "pozitiflerin_payi": sub["y"].sum() / df["y"].sum()})
    return pd.DataFrame(rows)


def investigation_predictability(df):
    """Ozellikler 'sorusturma numunesi mi?' sorusunu ne kadar cevapliyor?"""
    tmp = df.copy()
    tmp["y_true"] = tmp["y"]
    tmp["y"] = tmp["is_investigation"].astype(int)
    rows = []
    for fs in ("A", "B", "C", "CHLORINE_ONLY"):
        oof = run_cv(tmp, FEATURE_SETS[fs], "LR", SEEDS,
                     tune_threshold=False)
        m = mean_std(summarize(tmp, oof))
        rows.append({"feature_set": fs, "hedef": "is_investigation",
                     "ROC_AUC": m["ROC_AUC"], "ROC_AUC_std": m["ROC_AUC_std"]})
    return pd.DataFrame(rows)


def routine_eval(df, e01):
    routine = ~df["is_investigation"].values
    rows = []
    # a) tum veriyle egit, alt kumelerde degerlendir
    e01 = e01[e01["protocol"] == "purged"]
    for (fs, model), oof in e01.groupby(["feature_set", "model"]):
        for subset_name, subset in (("Tumu", None), ("Rutin", routine),
                                    ("Sorusturma", ~routine)):
            m = mean_std(summarize(df, oof, subset))
            rows.append({"egitim": "Tum veri", "test": subset_name,
                         "feature_set": fs, "model": model, **m})
    # b) yalniz rutinle egit, rutinde test et
    for fs in ("A", "C"):
        for model in MODEL_NAMES:
            oof = run_cv(df, FEATURE_SETS[fs], model, SEEDS,
                         train_mask=routine)
            m = mean_std(summarize(df, oof, routine))
            rows.append({"egitim": "Yalniz rutin", "test": "Rutin",
                         "feature_set": fs, "model": model, **m})
            print(f"rutin-egitim {fs} {model}: ROC={m['ROC_AUC']:.3f} "
                  f"PR={m['PR_AUC']:.3f}")
    return pd.DataFrame(rows)


def chlorine_scores(df):
    """Yuksek skor = yuksek risk. Eksik klor icin iki varsayim."""
    cl = df[CHLORINE]
    return {
        # eksik klor -> medyan (modelin yaptigi doldurmayla ayni)
        "klor_eksik_medyan": -cl.fillna(cl.median()).values,
        # eksik klor -> en riskli (klor olculmemis = en kotu durum)
        "klor_eksik_riskli": -cl.fillna(-1).values,
    }


def chlorine_baseline(df, e01):
    routine = ~df["is_investigation"].values
    rule = (df[CHLORINE].fillna(df[CHLORINE].median())
            <= LEGAL_LIMIT).astype(int).values
    lr = mean_prob(e01[(e01["protocol"] == "purged")
                       & (e01["feature_set"] == "A")
                       & (e01["model"] == "LR")]).sort_index().values
    scores = {**chlorine_scores(df), "model_LR_A": lr}

    rows, tests = [], []
    for subset_name, mask in (("Tumu", np.ones(len(df), bool)),
                              ("Rutin", routine)):
        y = df["y"].values[mask]
        for name, s in scores.items():
            m = compute_metrics(y, s[mask])
            m.pop("brier", None)
            rows.append({"test": subset_name, "skor": name, **m})
        rows.append({"test": subset_name,
                     "skor": f"kural_klor<={LEGAL_LIMIT}",
                     "MCC": matthews_corrcoef(y, rule[mask]),
                     "n": len(y), "n_pos": int(y.sum())})
        idx = np.where(mask)[0]
        for metric in ("ROC_AUC", "PR_AUC"):
            for ref in ("klor_eksik_medyan", "klor_eksik_riskli"):
                b = zape_bootstrap(df, idx, lr[idx], scores[ref][idx],
                                   metric=metric)
                tests.append({"test": subset_name, "metrik": metric,
                              "karsilastirma": f"model_LR_A - {ref}", **b})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    e01 = pd.read_csv(E01_OOF)

    s = sampling_structure(df)
    s.round(4).to_csv(OUT / "1_ornekleme_yapisi.csv", index=False)
    print(s.round(3).to_string(index=False))

    ip = investigation_predictability(df)
    ip.round(4).to_csv(OUT / "1b_sorusturma_tahmin_edilebilirligi.csv",
                       index=False)
    print(ip.round(3).to_string(index=False))

    r = routine_eval(df, e01)
    r.round(4).to_csv(OUT / "2_rutin_performans.csv", index=False)

    c, t = chlorine_baseline(df, e01)
    c.round(4).to_csv(OUT / "3_yalniz_klor.csv", index=False)
    t.round(4).to_csv(OUT / "3b_model_vs_klor_bootstrap.csv", index=False)
    print(c.round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
