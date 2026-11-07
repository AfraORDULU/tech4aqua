from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

INPUT = Path("results/oof_predictions.csv")
OUTPUT = Path("results/bootstrap_ci.csv")

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

rng = np.random.default_rng(RANDOM_STATE)

results = []

for scenario_name, scenario_df in df.groupby("Senaryo"):

    # Her ornegin 10 seed boyunca ortalama tahmin olasiligi
    averaged = scenario_df.groupby("idx", as_index=False).agg(
        probability=("p", "mean"),
        target=("y", "first"),
        zape=("zape", "first"),
    )

    # ZAPE bazli gruplama
    zape_groups = {
        name: group
        for name, group in averaged.groupby("zape")
    }

    zape_names = list(zape_groups.keys())

    roc_scores = []
    pr_scores = []

    for _ in range(N_BOOTSTRAP):

        sampled_zapes = rng.choice(
            zape_names,
            size=len(zape_names),
            replace=True,
        )

        sample = pd.concat(
            [zape_groups[zape] for zape in sampled_zapes],
            ignore_index=True,
        )

        # ROC-AUC iki sinif gerektirir
        if sample["target"].nunique() < 2:
            continue

        roc_scores.append(
            roc_auc_score(
                sample["target"],
                sample["probability"],
            )
        )

        pr_scores.append(
            average_precision_score(
                sample["target"],
                sample["probability"],
            )
        )

    results.append({
        "Senaryo": scenario_name,
        "ROC_AUC_Mean": np.mean(roc_scores),
        "ROC_AUC_CI_Lower": np.percentile(roc_scores, 2.5),
        "ROC_AUC_CI_Upper": np.percentile(roc_scores, 97.5),
        "PR_AUC_Mean": np.mean(pr_scores),
        "PR_AUC_CI_Lower": np.percentile(pr_scores, 2.5),
        "PR_AUC_CI_Upper": np.percentile(pr_scores, 97.5),
        "Valid_Bootstrap": len(roc_scores),
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT,
    index=False,
)

print("\n========== BOOTSTRAP %95 GUVEN ARALIKLARI ==========")

print(
    results_df.round(4).to_string(index=False)
)

print(f"\nSonuclar kaydedildi: {OUTPUT}")