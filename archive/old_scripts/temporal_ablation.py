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

TEMPORAL_FEATURES = [
    "PRECIP_LAG1_REGIONAL",
    "TEMP_LAG1_REGIONAL",
    "PRECIP_EWMA_REGIONAL",
    "TEMP_EWMA_REGIONAL",
]


# 1. Veriyi yükle
X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

scenarios = build_ablation_sets(X)


# 2. Mevcut F ve G senaryolarını al
base_scenarios = {
    "F_NoTemporal": scenarios["F_Genisletilmis"],
    "G_NoTemporal": scenarios["G_Bolgesiz"],
}


# 3. Dört meteorolojik geçmiş özelliğini çıkar
ablation_scenarios = {}

for name, X_scenario in base_scenarios.items():

    missing = [
        feature
        for feature in TEMPORAL_FEATURES
        if feature not in X_scenario.columns
    ]

    if missing:
        raise ValueError(
            f"{name} senaryosunda beklenen özellikler eksik: {missing}"
        )

    X_reduced = X_scenario.drop(
        columns=TEMPORAL_FEATURES
    )

    ablation_scenarios[name] = X_reduced

    print(
        f"{name}: "
        f"{X_scenario.shape[1]} -> "
        f"{X_reduced.shape[1]} özellik"
    )


# 4. Önceki deneyle aynı model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])


# 5. Tekrarlı grup bazlı çapraz doğrulama
results = []
oof_results = []

for scenario_name, X_scenario in ablation_scenarios.items():

    print(f"\n{scenario_name} basladi...")

    for seed in range(10):

        probabilities = oof_predictions(
            model,
            X_scenario,
            y,
            groups,
            seed=seed,
        )

        metrics = summarize(y, probabilities)

        results.append({
            "Senaryo": scenario_name,
            "Seed": seed,
            **metrics,
        })

        oof_results.append(pd.DataFrame({
            "Senaryo": scenario_name,
            "Seed": seed,
            "idx": range(len(y)),
            "zape": groups.to_numpy(),
            "y": y.to_numpy(),
            "p": probabilities,
        }))

        print(f"  Seed {seed} tamamlandi.")

    print(f"{scenario_name} tamamlandi.")


# 6. Yeni sonuçları ayrı dosyalara kaydet
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "temporal_ablation_results.csv",
    index=False,
)

pd.concat(
    oof_results,
    ignore_index=True
).to_csv(
    RESULTS_DIR / "temporal_ablation_oof.csv",
    index=False,
)


# 7. Önceki F ve G sonuçlarıyla karşılaştır
previous_path = RESULTS_DIR / "repeat_cv_results.csv"

if not previous_path.exists():
    raise FileNotFoundError(
        "Önceki deney sonuçları bulunamadı: "
        f"{previous_path}"
    )

previous_results = pd.read_csv(previous_path)

previous_results = previous_results[
    previous_results["Senaryo"].isin([
        "F_Genisletilmis",
        "G_Bolgesiz",
    ])
].copy()

expected_seeds = set(range(10))

for scenario_name in ["F_Genisletilmis", "G_Bolgesiz"]:
    scenario_seeds = set(
        previous_results.loc[
            previous_results["Senaryo"] == scenario_name,
            "Seed"
        ].astype(int)
    )

    if scenario_seeds != expected_seeds:
        raise ValueError(
            f"{scenario_name} için önceki deneyde "
            "0-9 arasındaki seed sonuçları eksik."
        )


# 8. Sonuçları birleştir
comparison = pd.concat(
    [previous_results, results_df],
    ignore_index=True
)

comparison.to_csv(
    RESULTS_DIR / "temporal_ablation_comparison.csv",
    index=False,
)


# 9. Özet tablo
summary = comparison.groupby("Senaryo")[
    ["ROC_AUC", "PR_AUC", "MCC"]
].agg(["mean", "std"])

print("\n========== METEOROLOJIK ABLATION SONUCLARI ==========")

print(summary.round(4).to_string())


# 10. Farkları hesapla
print("\n========== PERFORMANS FARKLARI ==========")

for base_name, reduced_name in [
    ("F_Genisletilmis", "F_NoTemporal"),
    ("G_Bolgesiz", "G_NoTemporal"),
]:

    base = comparison[
        comparison["Senaryo"] == base_name
    ].set_index("Seed")

    reduced = comparison[
        comparison["Senaryo"] == reduced_name
    ].set_index("Seed")

    print(f"\n{base_name} -> {reduced_name}")

    for metric in ["ROC_AUC", "PR_AUC", "MCC"]:

        differences = (
            reduced[metric] - base[metric]
        )

        print(
            f"{metric}: "
            f"ortalama fark = {differences.mean():+.4f}, "
            f"std = {differences.std():.4f}"
        )


print("\nDeney tamamlandi.")