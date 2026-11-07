# feature_formula_analysis.py

import numpy as np
import pandas as pd

from src.preprocessing import load_and_preprocess_data


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

CHLORINE = "Desinfetante residual"
TEMP = "TEMP MÁX (ºC)"
PRECIP = "PRECIPITAÇÃO (mm)"

DERIVED_FEATURES = [
    "DEGRADACAO_TERMICA_CLORO",
    "FLAG_CLORO_CRITICO",
    "FLAG_RUNOFF_RISCO",
    "FLAG_CALOR_EXTREMO"
]


def main():

    print("=" * 65)
    print("GOREV 24.1.2 - OZELLIK FORMUL ANALIZI")
    print("=" * 65)

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    selected = [
        CHLORINE,
        TEMP,
        PRECIP
    ] + DERIVED_FEATURES

    df = X[selected].copy()

    # ==================================================
    # 1. OZELLIK DAGILIMLARI
    # ==================================================

    print("\n1. TURETILMIS OZELLIK DAGILIMLARI")

    for feature in DERIVED_FEATURES:

        print("\n" + "-" * 50)
        print(feature)
        print("-" * 50)

        print("Eksik:", df[feature].isna().sum())

        print(
            df[feature].describe().to_string()
        )

        print(
            "Benzersiz deger sayisi:",
            df[feature].nunique(dropna=True)
        )

        if df[feature].nunique(dropna=True) <= 10:

            print("\nDeger dagilimi:")

            print(
                df[feature]
                .value_counts(dropna=False)
                .to_string()
            )

    # ==================================================
    # 2. KORELASYON ANALIZI
    # ==================================================

    print("\n" + "=" * 65)
    print("2. KORELASYON ANALIZI")
    print("=" * 65)

    correlation = df.corr(
        method="spearman"
    )

    print(
        correlation[
            DERIVED_FEATURES
        ].round(3).to_string()
    )

    # ==================================================
    # 3. EKSIK DEZENFEKTAN VE FLAG
    # ==================================================

    print("\n" + "=" * 65)
    print("3. EKSIK DEZENFEKTAN KONTROLU")
    print("=" * 65)

    missing_chlorine = df[
        df[CHLORINE].isna()
    ]

    print(
        "\nDezenfektan eksik gozlem:",
        len(missing_chlorine)
    )

    print(
        "\nFLAG_CLORO_CRITICO dagilimi:"
    )

    print(
        missing_chlorine["FLAG_CLORO_CRITICO"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ==================================================
    # 4. ORNEK GOZLEMLER
    # ==================================================

    print("\n" + "=" * 65)
    print("4. ILK 15 GOZLEM")
    print("=" * 65)

    print(
        df.head(15).to_string(index=False)
    )

    print("\nGOREV 24.1.2 TAMAMLANDI")


if __name__ == "__main__":
    main()