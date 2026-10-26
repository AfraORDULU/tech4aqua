
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_and_preprocess_data
from src.cold_start_ablation import build_ablation_sets
from src.evaluation import oof_predictions, summarize

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

X, y, groups, _ = load_and_preprocess_data(DATA_PATH)
scenarios = build_ablation_sets(X)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

results = []
oof_results = []

for scenario_name, X_scenario in scenarios.items():

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

    print(f"{scenario_name} tamamlandi.")

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    output_dir / "repeat_cv_results.csv",
    index=False,
)

pd.concat(oof_results, ignore_index=True).to_csv(
    output_dir / "oof_predictions.csv",
    index=False,
)

summary = results_df.groupby("Senaryo")[
    ["ROC_AUC", "PR_AUC", "MCC"]
].agg(["mean", "std"])

print("\n========== TEKRARLI CV OZETI ==========")
print(summary.round(4).to_string())