# missing_data_analysis.py

from pathlib import Path

import pandas as pd

from src.preprocessing import load_and_preprocess_data
from src.final_feature_sets import build_final_feature_sets


# ==================================================
# 1. AYARLAR
# ==================================================

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

OUTPUT_PATH = Path("results/missing_data_summary.csv")


# ==================================================
# 2. VERIYI YUKLE
# ==================================================

def main():

    print("=" * 65)
    print("GOREV 23.5.1 - EKSIK VERI ANALIZI")
    print("=" * 65)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

    scenarios = build_final_feature_sets(X)

    print(f"\nToplam gozlem: {len(X)}")
    print(f"Toplam ZAPE: {groups.nunique()}")
    print(f"Pozitif gozlem: {(y == 1).sum()}")
    print(f"Negatif gozlem: {(y == 0).sum()}")

    all_results = []

    # ==================================================
    # 3. SENARYO BAZLI EKSIKLIK
    # ==================================================

    for scenario_name, X_scenario in scenarios.items():

        print("\n" + "=" * 65)
        print(f"SENARYO: {scenario_name}")
        print("=" * 65)

        for feature in X_scenario.columns:

            missing_count = int(
                X_scenario[feature].isna().sum()
            )

            missing_percent = (
                missing_count / len(X_scenario)
            ) * 100

            all_results.append({
                "Scenario": scenario_name,
                "Feature": feature,
                "Missing_Count": missing_count,
                "Missing_Percent": missing_percent
            })

        scenario_results = pd.DataFrame(
            [
                row
                for row in all_results
                if row["Scenario"] == scenario_name
            ]
        )

        scenario_results = scenario_results.sort_values(
            "Missing_Percent",
            ascending=False
        )

        print(
            scenario_results[
                [
                    "Feature",
                    "Missing_Count",
                    "Missing_Percent"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

    # ==================================================
    # 4. SONUCLARI KAYDET
    # ==================================================

    results_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 65)
    print("SONUC")
    print("=" * 65)

    print(f"Toplam incelenen kayit: {len(results_df)}")

    print(
        "\nEksik veri bulunan ozellik-senaryo kayitlari:"
    )

    missing_features = results_df[
        results_df["Missing_Count"] > 0
    ]

    if missing_features.empty:

        print("Eksik veri bulunamadi.")

    else:

        print(
            missing_features[
                [
                    "Scenario",
                    "Feature",
                    "Missing_Count",
                    "Missing_Percent"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 23.5.1 TAMAMLANDI")


if __name__ == "__main__":
    main()