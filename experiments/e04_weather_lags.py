"""
Deney 4 - Gecikmeli hava ozellikleri (hocanin 4. maddesi).

Kontaminasyon genellikle numune gunundeki degil, onceki birkac gunun
yagisiyla iliskilidir. Open-Meteo gunluk verisinden numune gununden
ONCEKI gunler icin su ozellikler uretilir (src/weather.py):
  RAIN_3D, RAIN_7D, RAIN_14D : son 3 / 7 / 14 gunun toplam yagisi
  WET_DAYS_7D                : son 7 gunde >= 1 mm yagisli gun sayisi
  DAYS_SINCE_RAIN            : son yagisli gunden bu yana gecen gun
  TMEAN_7D                   : son 7 gunun ortalama sicakligi

Ozellik setleri mevcut setlere eklenerek karsilastirilir. Hem ZAPE gruplu
arindirilmis CV hem zamansal (rolling, gorulmemis ZAPE) protokol
kullanilir. Sonuclar tum numunelerde ve rutin numunelerde raporlanir.
LR icin "lag ekle - ekleme" farki ZAPE bootstrap ile test edilir.

On kosul: data/external/open_meteo_daily.csv (README'deki baglanti)
Calistirma: python -m experiments.e04_weather_lags
Ciktilar: results/e04_weather_lags/
"""

import pandas as pd

from src.config import RESULTS_DIR
from src.data import FEATURE_SETS, check_features, load_dataset
from src.evaluation import (mean_prob, mean_std, run_cv, run_temporal,
                            summarize, zape_bootstrap)
from src.models import MODEL_NAMES
from src.validation import temporal_splits
from src.weather import LAG_FEATURES, add_lag_features, same_day_agreement

OUT = RESULTS_DIR / "e04_weather_lags"
RAIN = ["RAIN_3D", "RAIN_7D"]

SETS = {
    "A": FEATURE_SETS["A"],
    "A+RAIN": FEATURE_SETS["A"] + RAIN,
    "A+LAG_ALL": FEATURE_SETS["A"] + LAG_FEATURES,
    "B": FEATURE_SETS["B"],
    "B+RAIN": FEATURE_SETS["B"] + RAIN,
    "LAG_ONLY": LAG_FEATURES + ["MES"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_lag_features(load_dataset())
    for cols in SETS.values():
        check_features(cols)

    agree = pd.DataFrame([same_day_agreement(df)])
    agree.round(3).to_csv(OUT / "0_open_meteo_uyum.csv", index=False)

    # Tek degiskenli tablo: pozitif / negatif medyanlari
    uni = df.groupby("y")[LAG_FEATURES].median().T
    uni.columns = ["negatif_medyan", "pozitif_medyan"]
    uni_r = (df[~df["is_investigation"]].groupby("y")[LAG_FEATURES]
             .median().T)
    uni["rutin_negatif_medyan"] = uni_r[0]
    uni["rutin_pozitif_medyan"] = uni_r[1]
    uni.round(2).to_csv(OUT / "1_tek_degiskenli.csv")
    print(uni.round(2))

    routine = ~df["is_investigation"].values
    temporal = list(temporal_splits(df, range(2015, 2026), unseen_zape=True))
    rows, lr_probs = [], {}
    for name, cols in SETS.items():
        for model in MODEL_NAMES:
            for protocol in ("zape_cv", "rolling_unseen"):
                if protocol == "zape_cv":
                    oof = run_cv(df, cols, model, (0, 1, 2, 3, 4))
                else:
                    oof = run_temporal(df, cols, model, temporal, (0, 1, 2))
                for subset_name, subset in (("Tumu", None),
                                            ("Rutin", routine)):
                    m = mean_std(summarize(df, oof, subset))
                    rows.append({"protocol": protocol, "feature_set": name,
                                 "model": model, "test": subset_name, **m})
                if model == "LR" and protocol == "zape_cv":
                    lr_probs[name] = mean_prob(oof).sort_index().values
                r = rows[-2]
                print(f"{protocol:14s} {name:9s} {model:4s} "
                      f"ROC={r['ROC_AUC']:.3f} PR={r['PR_AUC']:.3f} "
                      f"| rutin ROC={rows[-1]['ROC_AUC']:.3f} "
                      f"PR={rows[-1]['PR_AUC']:.3f}")

    pd.DataFrame(rows).round(4).to_csv(OUT / "2_summary.csv", index=False)

    # LR: gecikmeli ozellik eklemenin etkisi (eslestirilmis ZAPE bootstrap)
    tests = []
    idx_all = df.index.values
    idx_rut = df.index.values[routine]
    for new, base in (("A+RAIN", "A"), ("A+LAG_ALL", "A"),
                      ("B+RAIN", "B")):
        for subset_name, idx in (("Tumu", idx_all), ("Rutin", idx_rut)):
            for metric in ("ROC_AUC", "PR_AUC"):
                b = zape_bootstrap(df, idx, lr_probs[new][idx],
                                   lr_probs[base][idx], metric=metric)
                tests.append({"karsilastirma": f"{new} - {base}",
                              "test": subset_name, "metrik": metric, **b})
    t = pd.DataFrame(tests)
    t.round(4).to_csv(OUT / "3_bootstrap_LR.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
