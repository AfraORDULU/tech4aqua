# threshold_sensitivity_analysis.py

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


RESULTS_DIR = Path("results")

MODELS = [
    (
        "RandomForest",
        "A_Conservative",
        RESULTS_DIR / "final_rf_oof.csv"
    ),
    (
        "RandomForest",
        "C_Extended",
        RESULTS_DIR / "final_rf_oof.csv"
    ),
    (
        "LightGBM",
        "C_Extended",
        RESULTS_DIR / "final_lightgbm_oof.csv"
    )
]

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

EXPECTED_SEEDS = 10

OUTPUT_PATH = (
    RESULTS_DIR / "threshold_sensitivity_results.csv"
)


def prepare_predictions(path, scenario):

    df = pd.read_csv(path)

    df = df[
        df["Senaryo"] == scenario
    ].copy()

    if df.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario}"
        )

    if df.duplicated(["idx", "Seed"]).any():
        raise ValueError(
            f"Tekrarlanan tahmin kaydi: {path}"
        )

    seed_counts = df.groupby("idx")["Seed"].nunique()

    if not (seed_counts == EXPECTED_SEEDS).all():
        raise ValueError(
            f"Eksik seed tahminleri: {path}"
        )

    consistency = df.groupby("idx").agg(
        y_count=("y", "nunique"),
        zape_count=("zape", "nunique")
    )

    if (
        (consistency["y_count"] != 1).any()
        or (consistency["zape_count"] != 1).any()
    ):
        raise ValueError(
            f"Etiket veya ZAPE tutarsizligi: {path}"
        )

    return df.groupby(
        "idx",
        as_index=False
    ).agg(
        y=("y", "first"),
        p=("p", "mean")
    )


def main():

    print("=" * 60)
    print("GOREV 23.3 - ESIK DUYARLILIK ANALIZI")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Dosya zaten mevcut: {OUTPUT_PATH}"
        )

    results = []

    for model_name, scenario, path in MODELS:

        df = prepare_predictions(
            path,
            scenario
        )

        y_true = df["y"].to_numpy()
        probabilities = df["p"].to_numpy()

        print("\n" + "=" * 70)
        print(f"{model_name} - {scenario}")
        print("=" * 70)

        print(
            f"{'Esik':<8}"
            f"{'TP':<6}"
            f"{'FN':<6}"
            f"{'FP':<6}"
            f"{'Recall':<10}"
            f"{'Precision':<12}"
            f"{'MCC':<10}"
        )

        for threshold in THRESHOLDS:

            y_pred = (
                probabilities > threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            ).ravel()

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_true,
                y_pred
            )

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0
            )

            results.append({
                "Model": model_name,
                "Senaryo": scenario,
                "Threshold": threshold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "F1": f1,
                "MCC": mcc
            })

            print(
                f"{threshold:<8.2f}"
                f"{tp:<6}"
                f"{fn:<6}"
                f"{fp:<6}"
                f"{recall:<10.4f}"
                f"{precision:<12.4f}"
                f"{mcc:<10.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.3 TAMAMLANDI")


if __name__ == "__main__":
    main()