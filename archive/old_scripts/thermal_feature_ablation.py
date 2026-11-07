# thermal_feature_ablation.py

from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

from src.preprocessing import load_and_preprocess_data
from src.final_feature_sets import build_final_feature_sets
from src.evaluation import oof_predictions, summarize


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

OUTPUT_PATH = Path("results/thermal_feature_ablation.csv")

SEEDS = range(10)

REMOVED_FEATURE = "DEGRADACAO_TERMICA_CLORO"


def build_model():

    return Pipeline([
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
    ])


def main():

    print("=" * 65)
    print("GOREV 24.3 - TERMAL KLOR OZELLIGI ABLASYONU")
    print("=" * 65)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    feature_sets = build_final_feature_sets(X)

    X_full = feature_sets["C_Extended"]

    if REMOVED_FEATURE not in X_full.columns:
        raise ValueError(
            f"Ozellik bulunamadi: {REMOVED_FEATURE}"
        )

    X_reduced = X_full.drop(
        columns=[REMOVED_FEATURE]
    )

    scenarios = {
        "C_Extended": X_full,
        "C_NoThermalChlorine": X_reduced
    }

    results = []

    for scenario_name, X_scenario in scenarios.items():

        print("\n" + "=" * 65)
        print("SENARYO:", scenario_name)
        print("Ozellik sayisi:", X_scenario.shape[1])
        print("=" * 65)

        for seed in SEEDS:

            model = build_model()

            probabilities = oof_predictions(
                model=model,
                X=X_scenario,
                y=y,
                groups=groups,
                seed=seed,
                n_splits=5
            )

            metrics = summarize(
                y,
                probabilities,
                threshold=0.5
            )

            results.append({
                "Scenario": scenario_name,
                "Seed": seed,
                **metrics
            })

            print(
                f"Seed {seed:02d} | "
                f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
                f"PR-AUC: {metrics['PR_AUC']:.4f} | "
                f"MCC: {metrics['MCC']:.4f}"
            )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 65)
    print("ORTALAMA SONUCLAR")
    print("=" * 65)

    summary = results_df.groupby(
        "Scenario"
    )[
        ["ROC_AUC", "PR_AUC", "MCC"]
    ].agg(["mean", "std"])

    print(summary.round(4).to_string())

    print("\n" + "=" * 65)
    print("SENARYOLAR ARASI FARK")
    print("=" * 65)

    means = results_df.groupby(
        "Scenario"
    )[
        ["ROC_AUC", "PR_AUC", "MCC"]
    ].mean()

    difference = (
        means.loc["C_NoThermalChlorine"]
        - means.loc["C_Extended"]
    )

    print(difference.round(4).to_string())

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 24.3 TAMAMLANDI")


if __name__ == "__main__":
    main()