from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_and_preprocess_data
from src.cold_start_ablation import build_ablation_sets
from src.evaluation import oof_predictions, summarize


DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REMOVE_FEATURES = [
    "PRECIP_LAG1_REGIONAL",
    "TEMP_LAG1_REGIONAL",
    "PRECIP_EWMA_REGIONAL",
    "TEMP_EWMA_REGIONAL",
    "DELTA_DIAS_ZAPE",
    "LOG_DELTA_DIAS",
    "CLORO_RESIDUAL_PREV",
]


# 1. Veriyi yükle
X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

# 2. Mevcut senaryoları oluştur
scenarios = build_ablation_sets(X)

base_scenarios = {
    "F_StrictColdStart": scenarios["F_Genisletilmis"],
    "G_StrictColdStart": scenarios["G_Bolgesiz"],
}

strict_scenarios = {}

# 3. Geçmiş özelliklerini kaldır
for name, X_scenario in base_scenarios.items():

    missing = [
        feature
        for feature in REMOVE_FEATURES
        if feature not in X_scenario.columns
    ]

    if missing:
        raise ValueError(
            f"{name} icin beklenen ozellikler eksik: {missing}"
        )

    X_reduced = X_scenario.drop(
        columns=REMOVE_FEATURES
    )

    strict_scenarios[name] = X_reduced

    print(
        f"{name}: "
        f"{X_scenario.shape[1]} -> "
        f"{X_reduced.shape[1]} ozellik"
    )


# 4. Önceki deneylerle aynı model
model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])


results = []
oof_results = []


# 5. Tekrarlı grup bazlı çapraz doğrulama
for scenario_name, X_scenario in strict_scenarios.items():

    print(f"\n{scenario_name} basladi...")

    for seed in range(10):

        probabilities = oof_predictions(
            model,
            X_scenario,
            y,
            groups,
            seed=seed
        )

        metrics = summarize(
            y,
            probabilities
        )

        results.append({
            "Senaryo": scenario_name,
            "Seed": seed,
            **metrics
        })

        oof_results.append(
            pd.DataFrame({
                "Senaryo": scenario_name,
                "Seed": seed,
                "idx": range(len(y)),
                "zape": groups.to_numpy(),
                "y": y.to_numpy(),
                "p": probabilities
            })
        )

        print(
            f"  Seed {seed} tamamlandi."
        )

    print(
        f"{scenario_name} tamamlandi."
    )


# 6. Sonuçları ayrı dosyalara kaydet
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "strict_cold_start_results.csv",
    index=False
)

pd.concat(
    oof_results,
    ignore_index=True
).to_csv(
    RESULTS_DIR / "strict_cold_start_oof.csv",
    index=False
)


# 7. Sonuçları özetle
summary = (
    results_df
    .groupby("Senaryo")[
        ["ROC_AUC", "PR_AUC", "MCC"]
    ]
    .agg(["mean", "std"])
)

print(
    "\n========== STRICT COLD-START SONUCLARI =========="
)

print(
    summary.round(4).to_string()
)

print(
    "\nSonuclar kaydedildi:"
)

print(
    "results/strict_cold_start_results.csv"
)

print(
    "results/strict_cold_start_oof.csv"
)

print(
    "\nDeney tamamlandi."
)