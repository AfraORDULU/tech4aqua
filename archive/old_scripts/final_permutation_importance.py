# final_permutation_importance.py

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier

from src.preprocessing import load_and_preprocess_data
from src.final_feature_sets import build_final_feature_sets


# ==================================================
# 1. AYARLAR
# ==================================================

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

OUTPUT_PATH = Path("results/final_permutation_importance.csv")

N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]
N_REPEATS = 10

EXPERIMENTS = [
    ("RandomForest", "A_Conservative"),
    ("LightGBM", "C_Extended")
]


# ==================================================
# 2. MODELLER
# ==================================================

def build_model(model_name):

    if model_name == "RandomForest":

        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "LightGBM":

        classifier = LGBMClassifier(
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

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        ("classifier", classifier)
    ])


# ==================================================
# 3. ANA PROGRAM
# ==================================================

def main():

    print("=" * 65)
    print("GOREV 23.4.2 - PERMUTATION IMPORTANCE")
    print("=" * 65)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Dosya zaten mevcut: {OUTPUT_PATH}"
        )

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    scenarios = build_final_feature_sets(X)

    print(f"\nGozlem sayisi: {len(y)}")
    print(f"ZAPE sayisi: {groups.nunique()}")

    all_results = []

    for model_name, scenario in EXPERIMENTS:

        X_scenario = scenarios[scenario]

        print("\n" + "=" * 65)
        print(f"MODEL: {model_name} | SENARYO: {scenario}")
        print("=" * 65)

        for seed in SEEDS:

            cv = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, test_idx) in enumerate(
                cv.split(X_scenario, y, groups),
                start=1
            ):

                X_train = X_scenario.iloc[train_idx]
                X_test = X_scenario.iloc[test_idx]

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                # PR-AUC iki sinif gerektirir.
                if y_test.nunique() < 2:
                    print(
                        f"Seed {seed}, Fold {fold}: "
                        "tek sinifli test kati, atlandi."
                    )
                    continue

                model = build_model(model_name)

                model.fit(X_train, y_train)

                result = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    scoring="average_precision",
                    n_repeats=N_REPEATS,
                    random_state=42,
                    n_jobs=1
                )

                for feature, mean, std in zip(
                    X_scenario.columns,
                    result.importances_mean,
                    result.importances_std
                ):

                    all_results.append({
                        "Model": model_name,
                        "Senaryo": scenario,
                        "Seed": seed,
                        "Fold": fold,
                        "Feature": feature,
                        "Importance_Mean": mean,
                        "Importance_Std": std
                    })

                print(
                    f"Seed {seed:02d} | "
                    f"Fold {fold} tamamlandi."
                )

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        raise ValueError("Hicbir sonuc olusturulamadi.")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    summary = (
        results_df
        .groupby(
            ["Model", "Senaryo", "Feature"],
            as_index=False
        )
        .agg(
            Importance_Mean=("Importance_Mean", "mean"),
            Importance_SD_Across_Folds=("Importance_Mean", "std")
        )
    )

    for (model_name, scenario), group in summary.groupby(
        ["Model", "Senaryo"]
    ):

        print("\n" + "=" * 65)
        print(f"{model_name} - {scenario}")
        print("=" * 65)

        print(
            group.sort_values(
                "Importance_Mean",
                ascending=False
            )[
                [
                    "Feature",
                    "Importance_Mean",
                    "Importance_SD_Across_Folds"
                ]
            ].to_string(index=False)
        )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.4.2 TAMAMLANDI")


if __name__ == "__main__":
    main()