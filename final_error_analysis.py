# final_error_analysis.py

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

OUTPUT_PATH = RESULTS_DIR / "final_error_analysis.csv"

THRESHOLD = 0.5
EXPECTED_SEEDS = 10


def prepare_predictions(path, scenario):

    df = pd.read_csv(path)

    df = df[df["Senaryo"] == scenario].copy()

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

    result = df.groupby("idx", as_index=False).agg(
        y=("y", "first"),
        zape=("zape", "first"),
        p=("p", "mean")
    )

    if result["p"].isna().any():
        raise ValueError(
            f"Eksik tahmin: {path}"
        )

    return result


def main():

    print("=" * 60)
    print("GOREV 23.1 - HATA ANALIZI")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    results = []

    for model_name, scenario, path in MODELS:

        df = prepare_predictions(path, scenario)

        y_true = df["y"].to_numpy()

        y_pred = (
            df["p"].to_numpy() > THRESHOLD
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

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
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

        results.append({
            "Model": model_name,
            "Senaryo": scenario,
            "Threshold": THRESHOLD,
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

        print("\n" + "-" * 60)
        print(f"{model_name} - {scenario}")
        print("-" * 60)

        print(f"TN: {tn}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"TP: {tp}")

        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1:          {f1:.4f}")
        print(f"MCC:         {mcc:.4f}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuclar kaydedildi:", OUTPUT_PATH)
    print("\nGOREV 23.1 TAMAMLANDI")


if __name__ == "__main__":
    main()