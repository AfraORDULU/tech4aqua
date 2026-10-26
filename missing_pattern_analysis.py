# missing_pattern_analysis.py

from pathlib import Path

import pandas as pd

from src.preprocessing import load_and_preprocess_data


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

OUTPUT_PATH = Path("results/missing_pattern_analysis.csv")

CHLORINE = "Desinfetante residual"
DERIVED = "DEGRADACAO_TERMICA_CLORO"
FLAG = "FLAG_CLORO_CRITICO"


def main():

    print("=" * 65)
    print("GOREV 23.5.2 - EKSIKLIK DESENI ANALIZI")
    print("=" * 65)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    required = [CHLORINE, DERIVED, FLAG]

    missing_columns = [
        col for col in required
        if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Eksik sutunlar: {missing_columns}"
        )

    df = X[required].copy()

    # Pozisyon bazli eslestirme:
    # X, y ve groups ayni gozlem sirasinda olmali.
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("Gozlem sayilari uyusmuyor.")

    df["y"] = y.to_numpy()
    df["zape"] = groups.to_numpy()

    df["Chlorine_Missing"] = df[CHLORINE].isna()
    df["Derived_Missing"] = df[DERIVED].isna()

    # ==================================================
    # 1. EKSIKLIKLER AYNI GOZLEMLERDE MI?
    # ==================================================

    print("\n1. EKSIKLIK KARSILASTIRMASI")

    print(
        "\nDezenfektan eksik:",
        int(df["Chlorine_Missing"].sum())
    )

    print(
        "Turetilmis ozellik eksik:",
        int(df["Derived_Missing"].sum())
    )

    both_missing = (
        df["Chlorine_Missing"]
        & df["Derived_Missing"]
    ).sum()

    print(
        "Ikisi de eksik:",
        int(both_missing)
    )

    same_pattern = (
        df["Chlorine_Missing"]
        == df["Derived_Missing"]
    ).all()

    print(
        "Eksiklik desenleri tamamen ayni mi?",
        bool(same_pattern)
    )

    # ==================================================
    # 2. HEDEF SINIFINA GORE EKSIKLIK
    # ==================================================

    print("\n2. HEDEF SINIFINA GORE EKSIKLIK")

    class_summary = df.groupby("y").agg(
        Total=("y", "size"),
        Chlorine_Missing=("Chlorine_Missing", "sum"),
        Derived_Missing=("Derived_Missing", "sum")
    )

    class_summary["Chlorine_Missing_Pct"] = (
        100
        * class_summary["Chlorine_Missing"]
        / class_summary["Total"]
    )

    print(class_summary.to_string())

    # ==================================================
    # 3. ZAPE BAZLI EKSIKLIK
    # ==================================================

    print("\n3. ZAPE BAZLI EKSIKLIK")

    zape_summary = df.groupby(
        "zape",
        as_index=False
    ).agg(
        Total=("y", "size"),
        Positive=("y", "sum"),
        Chlorine_Missing=("Chlorine_Missing", "sum"),
        Derived_Missing=("Derived_Missing", "sum")
    )

    zape_summary["Chlorine_Missing_Pct"] = (
        100
        * zape_summary["Chlorine_Missing"]
        / zape_summary["Total"]
    )

    affected_zapes = zape_summary[
        zape_summary["Chlorine_Missing"] > 0
    ].sort_values(
        ["Chlorine_Missing", "Chlorine_Missing_Pct"],
        ascending=False
    )

    if affected_zapes.empty:
        print("Eksik dezenfektan olcumu bulunan ZAPE yok.")
    else:
        print(affected_zapes.to_string(index=False))

    # ==================================================
    # 4. FLAG DAVRANISI
    # ==================================================

    print("\n4. DEZENFEKTAN EKSIKKEN FLAG DEGERLERI")

    flag_summary = (
        df.loc[df["Chlorine_Missing"], FLAG]
        .value_counts(dropna=False)
    )

    print(flag_summary.to_string())

    # ==================================================
    # 5. KAYDET
    # ==================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    zape_summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.5.2 TAMAMLANDI")


if __name__ == "__main__":
    main()