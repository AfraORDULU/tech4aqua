from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


RESULTS_DIR = Path("results")

OLD_PATH = RESULTS_DIR / "oof_predictions.csv"
NEW_PATH = RESULTS_DIR / "temporal_ablation_oof.csv"

OUTPUT_PATH = RESULTS_DIR / "paired_temporal_bootstrap.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

COMPARISONS = [
    ("F_Genisletilmis", "F_NoTemporal"),
    ("G_Bolgesiz", "G_NoTemporal"),
]


def prepare_predictions(df, scenario_name):

    subset = df.loc[
        df["Senaryo"] == scenario_name
    ].copy()

    if subset.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario_name}"
        )

    # Her gözlem için 10 seed tahmininin ortalaması
    grouped = (
        subset.groupby("idx", as_index=False)
        .agg(
            zape=("zape", "first"),
            y=("y", "first"),
            p=("p", "mean"),
            seed_count=("Seed", "nunique"),
        )
    )

    if not (grouped["seed_count"] == 10).all():
        raise ValueError(
            f"{scenario_name}: Her gozlem icin 10 seed bulunmuyor."
        )

    return grouped[["idx", "zape", "y", "p"]]


def calculate_metrics(y, probabilities):

    return {
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(y, probabilities),
    }


def bootstrap_comparison(base_df, reduced_df, rng):

    merged = base_df.merge(
        reduced_df,
        on="idx",
        suffixes=("_base", "_reduced"),
        validate="one_to_one",
    )

    # Aynı gözlemler ve aynı hedefler kullanılmalı
    if not (
        merged["y_base"].to_numpy()
        == merged["y_reduced"].to_numpy()
    ).all():
        raise ValueError("Hedef degerleri uyusmuyor.")

    if not (
        merged["zape_base"].to_numpy()
        == merged["zape_reduced"].to_numpy()
    ).all():
        raise ValueError("ZAPE gruplari uyusmuyor.")

    y = merged["y_base"].to_numpy()
    p_base = merged["p_base"].to_numpy()
    p_reduced = merged["p_reduced"].to_numpy()

    # Orijinal veri üzerindeki fark
    original_base = calculate_metrics(y, p_base)
    original_reduced = calculate_metrics(y, p_reduced)

    original_diff = {
        metric: original_reduced[metric] - original_base[metric]
        for metric in ["ROC_AUC", "PR_AUC"]
    }

    # ZAPE bazında bootstrap
    groups = merged["zape_base"].to_numpy()
    unique_groups = np.unique(groups)

    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }

    bootstrap_diffs = {
        "ROC_AUC": [],
        "PR_AUC": [],
    }

    skipped = 0

    for _ in range(N_BOOTSTRAP):

        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        sampled_indices = np.concatenate([
            group_indices[group]
            for group in sampled_groups
        ])

        sampled_y = y[sampled_indices]

        # ROC-AUC için iki sınıf da bulunmalı
        if len(np.unique(sampled_y)) < 2:
            skipped += 1
            continue

        base_metrics = calculate_metrics(
            sampled_y,
            p_base[sampled_indices],
        )

        reduced_metrics = calculate_metrics(
            sampled_y,
            p_reduced[sampled_indices],
        )

        for metric in bootstrap_diffs:

            difference = (
                reduced_metrics[metric]
                - base_metrics[metric]
            )

            bootstrap_diffs[metric].append(difference)

    rows = []

    for metric, differences in bootstrap_diffs.items():

        differences = np.asarray(differences)

        if len(differences) == 0:
            raise ValueError(
                "Gecerli bootstrap orneklemi bulunamadi."
            )

        lower, upper = np.percentile(
            differences,
            [2.5, 97.5],
        )

        rows.append({
            "Metric": metric,
            "Base_Score": original_base[metric],
            "Reduced_Score": original_reduced[metric],
            "Difference": original_diff[metric],
            "CI_Lower": lower,
            "CI_Upper": upper,
            "Valid_Bootstrap": len(differences),
            "Skipped_Bootstrap": skipped,
        })

    return rows


def main():

    old_df = pd.read_csv(OLD_PATH)
    new_df = pd.read_csv(NEW_PATH)

    rng = np.random.default_rng(RANDOM_STATE)

    all_results = []

    print("\n========== PAIRED ZAPE BOOTSTRAP ==========")

    for base_name, reduced_name in COMPARISONS:

        print(f"\n{base_name} -> {reduced_name}")

        base = prepare_predictions(
            old_df,
            base_name,
        )

        reduced = prepare_predictions(
            new_df,
            reduced_name,
        )

        results = bootstrap_comparison(
            base,
            reduced,
            rng,
        )

        for row in results:

            row["Base_Scenario"] = base_name
            row["Reduced_Scenario"] = reduced_name

            all_results.append(row)

            print(
                f"{row['Metric']}: "
                f"fark={row['Difference']:+.4f}, "
                f"%95 GA=[{row['CI_Lower']:+.4f}, "
                f"{row['CI_Upper']:+.4f}]"
            )

    result_df = pd.DataFrame(all_results)

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSonuclar kaydedildi: {OUTPUT_PATH}")
    print("\nAnaliz tamamlandi.")


if __name__ == "__main__":
    main()