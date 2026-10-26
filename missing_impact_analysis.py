# missing_impact_analysis.py

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import load_and_preprocess_data


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

RESULTS_DIR = Path("results")

OUTPUT_PATH = RESULTS_DIR / "missing_impact_analysis.csv"

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

THRESHOLD = 0.5
EXPECTED_SEEDS = 10

CHLORINE = "Desinfetante residual"


def load_predictions(path, scenario):

    df = pd.read_csv(path)

    required = {
        "Senaryo", "Seed", "idx", "zape", "y", "p"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Eksik sutunlar: {path}"
        )

    df = df[df["Senaryo"] == scenario].copy()

    if df.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario}"
        )

    if df.duplicated(["idx", "Seed"]).any():
        raise ValueError(
            f"Tekrarlanan idx/Seed: {path}"
        )

    counts = df.groupby("idx")["Seed"].nunique()

    if not (counts == EXPECTED_SEEDS).all():
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
        zape=("zape", "first"),
        y=("y", "first"),
        p=("p", "mean")
    )


def main():

    print("=" * 65)
    print("GOREV 23.5.3 - EKSIK VERI ETKISI")
    print("=" * 65)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    # OOF dosyalarindaki idx, X'in satir pozisyonu
    # olarak kaydedilmisti.
    reference = pd.DataFrame({
        "idx": np.arange(len(X)),
        "zape_reference": groups.to_numpy(),
        "y_reference": y.to_numpy(),
        "Chlorine_Missing": X[CHLORINE].isna().to_numpy()
    })

    all_results = []

    for model_name, scenario, path in MODELS:

        predictions = load_predictions(
            path,
            scenario
        )

        df = predictions.merge(
            reference,
            on="idx",
            validate="one_to_one"
        )

        if len(df) != len(reference):
            raise ValueError(
                "OOF ve referans gozlem sayilari farkli."
            )

        if not (
            df["y"].to_numpy()
            == df["y_reference"].to_numpy()
        ).all():
            raise ValueError(
                "Hedef degerleri eslesmiyor."
            )

        if not (
            df["zape"].astype(str).to_numpy()
            == df["zape_reference"].astype(str).to_numpy()
        ).all():
            raise ValueError(
                "ZAPE degerleri eslesmiyor."
            )

        df["prediction"] = (
            df["p"] > THRESHOLD
        ).astype(int)

        print("\n" + "=" * 65)
        print(f"{model_name} - {scenario}")
        print("=" * 65)

        for missing_status, label in [
            (True, "Eksik olcum"),
            (False, "Olcum mevcut")
        ]:

            subset = df[
                df["Chlorine_Missing"] == missing_status
            ]

            positives = subset[subset["y"] == 1]
            negatives = subset[subset["y"] == 0]

            tp = int(
                (positives["prediction"] == 1).sum()
            )

            fn = int(
                (positives["prediction"] == 0).sum()
            )

            fp = int(
                (negatives["prediction"] == 1).sum()
            )

            tn = int(
                (negatives["prediction"] == 0).sum()
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else np.nan
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else np.nan
            )

            all_results.append({
                "Model": model_name,
                "Senaryo": scenario,
                "Missing_Group": label,
                "N": len(subset),
                "Positive": len(positives),
                "Negative": len(negatives),
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "Recall": recall,
                "Precision": precision
            })

            print(f"\n{label}")
            print(f"Gozlem: {len(subset)}")
            print(f"Pozitif: {len(positives)}")
            print(f"Negatif: {len(negatives)}")
            print(f"TP: {tp} | FN: {fn}")
            print(f"FP: {fp} | TN: {tn}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.5.3 TAMAMLANDI")


if __name__ == "__main__":
    main()