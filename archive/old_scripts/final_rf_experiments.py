# final_rf_experiments.py

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing import load_and_preprocess_data
from src.final_feature_sets import build_final_feature_sets
from src.evaluation import oof_predictions, summarize


# ==================================================
# 1. DOSYA YOLLARI
# ==================================================

DATA_PATH = (
    "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"
)

RESULTS_DIR = Path("results")

RESULTS_PATH = RESULTS_DIR / "final_rf_results.csv"
OOF_PATH = RESULTS_DIR / "final_rf_oof.csv"


# ==================================================
# 2. DENEY AYARLARI
# ==================================================

SEEDS = list(range(10))

N_SPLITS = 5


# ==================================================
# 3. RANDOM FOREST MODELI
# ==================================================

def build_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )


# ==================================================
# 4. ANA DENEY
# ==================================================

def main():

    print("=" * 60)
    print("GOREV 21.3 - FINAL RANDOM FOREST DENEYLERI")
    print("=" * 60)

    # Veri setini yukle
    X, y, groups, _ = load_and_preprocess_data(
        DATA_PATH
    )

    # Aday ozellik senaryolarini olustur
    scenarios = build_final_feature_sets(X)

    print("\nGozlem sayisi:", len(y))
    print("ZAPE sayisi:", groups.nunique())

    print("\nSenaryolar:")

    for name, data in scenarios.items():
        print(f"  {name}: {data.shape[1]} ozellik")

    # Sonuclari saklamak icin
    results = []
    oof_records = []

    # Her senaryo icin
    for scenario_name, X_scenario in scenarios.items():

        print("\n" + "=" * 60)
        print("SENARYO:", scenario_name)
        print("=" * 60)

        # Her seed icin
        for seed in SEEDS:

            model = build_model()

            # ZAPE bazli OOF tahminleri
            probabilities = oof_predictions(
                model=model,
                X=X_scenario,
                y=y,
                groups=groups,
                seed=seed,
                n_splits=N_SPLITS
            )

            # Performans metrikleri
            metrics = summarize(
                y,
                probabilities
            )

            # Sonuc kaydi
            results.append({
                "Senaryo": scenario_name,
                "Seed": seed,
                "Ozellik_Sayisi": X_scenario.shape[1],
                **metrics
            })

            # OOF tahmin kayitlari
            for idx in range(len(y)):

                oof_records.append({
                    "Senaryo": scenario_name,
                    "Seed": seed,
                    "idx": idx,
                    "zape": groups.iloc[idx],
                    "y": int(y.iloc[idx]),
                    "p": float(probabilities[idx])
                })

            print(
                f"Seed {seed:02d} | "
                f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
                f"PR-AUC: {metrics['PR_AUC']:.4f} | "
                f"MCC: {metrics['MCC']:.4f}"
            )

    # ==================================================
    # 5. SONUCLARI KAYDET
    # ==================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(results)
    oof_df = pd.DataFrame(oof_records)

    # Eski deney dosyalarini koru
    for path in [RESULTS_PATH, OOF_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Dosya zaten mevcut, uzerine yazilmadi: {path}"
            )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    oof_df.to_csv(
        OOF_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("ORTALAMA PERFORMANS SONUCLARI")
    print("=" * 60)

    summary = results_df.groupby("Senaryo")[
        ["ROC_AUC", "PR_AUC", "MCC"]
    ].agg(["mean", "std"])

    print(summary.round(4).to_string())

    print("\nKaydedilen dosyalar:")
    print(" -", RESULTS_PATH)
    print(" -", OOF_PATH)

    print("\nGOREV 21.3 TAMAMLANDI")


if __name__ == "__main__":
    main()