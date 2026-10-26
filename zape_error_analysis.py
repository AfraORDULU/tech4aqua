# zape_error_analysis.py

from pathlib import Path

import pandas as pd


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

OUTPUT_PATH = RESULTS_DIR / "zape_error_analysis.csv"

THRESHOLD = 0.5
EXPECTED_SEEDS = 10


def prepare_predictions(path, scenario):

    df = pd.read_csv(path)

    required = {
        "Senaryo", "Seed", "idx", "zape", "y", "p"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Eksik sutunlar: {path}"
        )

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

    result = df.groupby(
        "idx",
        as_index=False
    ).agg(
        zape=("zape", "first"),
        y=("y", "first"),
        p=("p", "mean")
    )

    if result["p"].isna().any():
        raise ValueError(
            f"Eksik tahmin: {path}"
        )

    return result


def main():

    print("=" * 60)
    print("GOREV 23.2 - ZAPE HATA ANALIZI")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Dosya zaten mevcut: {OUTPUT_PATH}"
        )

    all_results = []

    for model_name, scenario, path in MODELS:

        df = prepare_predictions(
            path,
            scenario
        )

        df["prediction"] = (
            df["p"] > THRESHOLD
        ).astype(int)

        df["TP"] = (
            (df["y"] == 1)
            & (df["prediction"] == 1)
        ).astype(int)

        df["TN"] = (
            (df["y"] == 0)
            & (df["prediction"] == 0)
        ).astype(int)

        df["FP"] = (
            (df["y"] == 0)
            & (df["prediction"] == 1)
        ).astype(int)

        df["FN"] = (
            (df["y"] == 1)
            & (df["prediction"] == 0)
        ).astype(int)

        grouped = df.groupby(
            "zape",
            as_index=False
        ).agg(
            Gozlem=("y", "size"),
            Pozitif=("y", "sum"),
            TP=("TP", "sum"),
            TN=("TN", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum")
        )

        grouped["Negatif"] = (
            grouped["Gozlem"]
            - grouped["Pozitif"]
        )

        # Paydasi sifir olan bolgelerde
        # oran tanimsiz birakilir.
        grouped["Recall"] = (
            grouped["TP"]
            / grouped["Pozitif"].replace(0, float("nan"))
        )

        grouped["Specificity"] = (
            grouped["TN"]
            / grouped["Negatif"].replace(0, float("nan"))
        )

        grouped["Model"] = model_name
        grouped["Senaryo"] = scenario

        all_results.append(grouped)

        print("\n" + "=" * 60)
        print(f"{model_name} - {scenario}")
        print("=" * 60)

        print("\nEn fazla yanlis negatif olan ZAPE'ler:")

        top_fn = grouped.sort_values(
            ["FN", "Pozitif"],
            ascending=False
        ).head(10)

        print(
            top_fn[
                ["zape", "Gozlem", "Pozitif", "FN", "Recall"]
            ].to_string(index=False)
        )

        print("\nEn fazla yanlis pozitif olan ZAPE'ler:")

        top_fp = grouped.sort_values(
            ["FP", "Negatif"],
            ascending=False
        ).head(10)

        print(
            top_fp[
                ["zape", "Gozlem", "Negatif", "FP", "Specificity"]
            ].to_string(index=False)
        )

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.2 TAMAMLANDI")


if __name__ == "__main__":
    main()