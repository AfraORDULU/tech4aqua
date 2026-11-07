# final_lightgbm_experiments.py

from pathlib import Path

import pandas as pd

from lightgbm import LGBMClassifier

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

RESULTS_PATH = RESULTS_DIR / "final_lightgbm_results.csv"
OOF_PATH = RESULTS_DIR / "final_lightgbm_oof.csv"


# ==================================================
# 2. DENEY AYARLARI
# ==================================================

SEEDS = list(range(10))
N_SPLITS = 5


# ==================================================
# 3. LIGHTGBM MODELI
# ==================================================

def build_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            ),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=7,
                    max_depth=4,
                    min_child_samples=15,
                    reg_lambda=1.0,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                    verbosity=-1
                )
            )
        ]
    )


# ==================================================
# 4. ANA DENEY
# ==================================================

def main():

    print("=" * 60)
    print("GOREV 22.2 - LIGHTGBM DENEYLERI")
    print("=" * 60)

    # Mevcut veri hazirlama sistemini kullan
    X, y, groups, _ = load_and_preprocess_data(
        DATA_PATH
    )

    scenarios = build_final_feature_sets(X)

    print("\nGozlem sayisi:", len(y))
    print("ZAPE sayisi:", groups.nunique())

    print("\nSenaryolar:")
    for name, data in scenarios.items():
        print(f"  {name}: {data.shape[1]} ozellik")

    # Mevcut sonuc dosyalarini koru
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in [RESULTS_PATH, OOF_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Dosya zaten mevcut: {path}"
            )

    results = []
    oof_records = []

    for scenario_name, X_scenario in scenarios.items():

        print("\n" + "=" * 60)
        print("SENARYO:", scenario_name)
        print("=" * 60)

        for seed in SEEDS:

            model = build_model()

            probabilities = oof_predictions(
                model=model,
                X=X_scenario,
                y=y,
                groups=groups,
                seed=seed,
                n_splits=N_SPLITS
            )

            metrics = summarize(
                y,
                probabilities
            )

            results.append({
                "Senaryo": scenario_name,
                "Seed": seed,
                "Ozellik_Sayisi": X_scenario.shape[1],
                **metrics
            })

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

    results_df = pd.DataFrame(results)
    oof_df = pd.DataFrame(oof_records)

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    oof_df.to_csv(
        OOF_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("ORTALAMA LIGHTGBM SONUCLARI")
    print("=" * 60)

    summary = results_df.groupby("Senaryo")[
        ["ROC_AUC", "PR_AUC", "MCC"]
    ].agg(["mean", "std"])

    print(summary.round(4).to_string())

    print("\nKaydedilen dosyalar:")
    print(" -", RESULTS_PATH)
    print(" -", OOF_PATH)

    print("\nGOREV 22.2 TAMAMLANDI")


if __name__ == "__main__":
    main()