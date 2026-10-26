# paired_final_bootstrap.py

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ==================================================
# 1. AYARLAR
# ==================================================

RESULTS_DIR = Path("results")

INPUT_PATH = RESULTS_DIR / "final_rf_oof.csv"
OUTPUT_PATH = RESULTS_DIR / "paired_final_bootstrap.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42
EXPECTED_SEEDS = 10

COMPARISONS = [
    ("A_Conservative", "C_Extended"),
    ("B_Meteorological", "A_Conservative"),
    ("B_Meteorological", "C_Extended"),
]


# ==================================================
# 2. OOF TAHMINLERINI HAZIRLAMA
# ==================================================

def prepare_predictions(df, scenario_name):

    subset = df[
        df["Senaryo"] == scenario_name
    ].copy()

    if subset.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario_name}"
        )

    seed_counts = subset.groupby("idx")["Seed"].nunique()

    if not (seed_counts == EXPECTED_SEEDS).all():
        raise ValueError(
            f"{scenario_name}: Eksik seed tahminleri var."
        )

    consistency = subset.groupby("idx").agg(
        y_count=("y", "nunique"),
        zape_count=("zape", "nunique")
    )

    if (
        (consistency["y_count"] != 1).any()
        or (consistency["zape_count"] != 1).any()
    ):
        raise ValueError(
            f"{scenario_name}: y veya ZAPE tutarsiz."
        )

    prepared = subset.groupby("idx", as_index=False).agg(
        zape=("zape", "first"),
        y=("y", "first"),
        p=("p", "mean")
    )

    return prepared


# ==================================================
# 3. METRIKLER
# ==================================================

def calculate_metrics(y, probabilities):

    return {
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(
            y,
            probabilities
        )
    }


# ==================================================
# 4. ESLESTIRILMIS ZAPE BOOTSTRAP
# ==================================================

def bootstrap_comparison(
    base,
    candidate,
    base_name,
    candidate_name,
    rng
):

    merged = base.merge(
        candidate,
        on="idx",
        suffixes=("_base", "_candidate"),
        validate="one_to_one"
    )

    if len(merged) != len(base) or len(merged) != len(candidate):
        raise ValueError("Senaryolarin gozlem sayilari farkli.")

    if not (
        merged["y_base"].to_numpy()
        == merged["y_candidate"].to_numpy()
    ).all():
        raise ValueError("Hedef degerleri uyusmuyor.")

    if not (
        merged["zape_base"].to_numpy()
        == merged["zape_candidate"].to_numpy()
    ).all():
        raise ValueError("ZAPE bilgileri uyusmuyor.")

    y = merged["y_base"].to_numpy()

    p_base = merged["p_base"].to_numpy()
    p_candidate = merged["p_candidate"].to_numpy()

    zape = merged["zape_base"].to_numpy()

    base_metrics = calculate_metrics(y, p_base)
    candidate_metrics = calculate_metrics(y, p_candidate)

    unique_zapes = np.unique(zape)

    group_indices = {
        group: np.flatnonzero(zape == group)
        for group in unique_zapes
    }

    differences = {
        "ROC_AUC": [],
        "PR_AUC": []
    }

    skipped = 0

    for _ in range(N_BOOTSTRAP):

        sampled_zapes = rng.choice(
            unique_zapes,
            size=len(unique_zapes),
            replace=True
        )

        sampled_indices = np.concatenate([
            group_indices[group]
            for group in sampled_zapes
        ])

        y_sample = y[sampled_indices]

        if len(np.unique(y_sample)) < 2:
            skipped += 1
            continue

        base_sample = calculate_metrics(
            y_sample,
            p_base[sampled_indices]
        )

        candidate_sample = calculate_metrics(
            y_sample,
            p_candidate[sampled_indices]
        )

        for metric in differences:
            differences[metric].append(
                candidate_sample[metric]
                - base_sample[metric]
            )

    rows = []

    for metric, values in differences.items():

        values = np.asarray(values)

        if len(values) == 0:
            raise ValueError(
                "Gecerli bootstrap ornegi olusturulamadi."
            )

        lower, upper = np.percentile(
            values,
            [2.5, 97.5]
        )

        rows.append({
            "Base": base_name,
            "Candidate": candidate_name,
            "Metric": metric,
            "Base_Score": base_metrics[metric],
            "Candidate_Score": candidate_metrics[metric],
            "Difference": (
                candidate_metrics[metric]
                - base_metrics[metric]
            ),
            "CI_Lower": lower,
            "CI_Upper": upper,
            "Valid_Bootstrap": len(values),
            "Skipped_Bootstrap": skipped
        })

    return rows


# ==================================================
# 5. ANA PROGRAM
# ==================================================

def main():

    print("=" * 60)
    print("GOREV 21.4 - FINAL BOOTSTRAP")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"OOF dosyasi bulunamadi: {INPUT_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "Senaryo", "Seed", "idx", "zape", "y", "p"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "OOF dosyasinda gerekli sutunlar eksik."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    all_results = []

    for base_name, candidate_name in COMPARISONS:

        base = prepare_predictions(df, base_name)

        candidate = prepare_predictions(
            df,
            candidate_name
        )

        rows = bootstrap_comparison(
            base,
            candidate,
            base_name,
            candidate_name,
            rng
        )

        all_results.extend(rows)

        print(
            f"\n{base_name} -> {candidate_name}"
        )

        for row in rows:
            print(
                f"{row['Metric']}: "
                f"fark={row['Difference']:+.4f}, "
                f"%95 GA=["
                f"{row['CI_Lower']:+.4f}, "
                f"{row['CI_Upper']:+.4f}]"
            )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSonuclar kaydedildi:", OUTPUT_PATH)
    print("\nGOREV 21.4 TAMAMLANDI")


if __name__ == "__main__":
    main()