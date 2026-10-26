# model_bootstrap_comparison.py

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


# ==================================================
# 1. AYARLAR
# ==================================================

RESULTS_DIR = Path("results")

MODEL_FILES = {
    "RandomForest": RESULTS_DIR / "final_rf_oof.csv",
    "LightGBM": RESULTS_DIR / "final_lightgbm_oof.csv",
    "LogisticRegression": RESULTS_DIR / "final_logistic_oof.csv"
}

SCENARIOS = [
    "A_Conservative",
    "C_Extended"
]

OUTPUT_PATH = RESULTS_DIR / "model_bootstrap_comparison.csv"

N_BOOTSTRAP = 2000
EXPECTED_SEEDS = 10
RANDOM_STATE = 42


# ==================================================
# 2. VERI HAZIRLAMA
# ==================================================

def load_model_predictions(path, scenario):

    df = pd.read_csv(path)

    required = {
        "Senaryo", "Seed", "idx", "zape", "y", "p"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Eksik sutunlar: {path}"
        )

    df = df[df["Senaryo"] == scenario].copy()

    if df.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario}"
        )

    if df.duplicated(["idx", "Seed"]).any():
        raise ValueError(
            f"Tekrarlanan idx/Seed kaydi: {path}"
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

    result = df.groupby("idx", as_index=False).agg(
        zape=("zape", "first"),
        y=("y", "first"),
        p=("p", "mean")
    )

    if result["p"].isna().any():
        raise ValueError(
            f"Eksik tahmin olasiligi: {path}"
        )

    return result


# ==================================================
# 3. METRIK HESAPLAMA
# ==================================================

def calculate_metrics(y, p):

    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p)
    }


# ==================================================
# 4. ESLESTIRILMIS BOOTSTRAP
# ==================================================

def compare_models(
    model_a,
    model_b,
    name_a,
    name_b,
    scenario,
    rng
):

    merged = model_a.merge(
        model_b,
        on="idx",
        suffixes=("_a", "_b"),
        validate="one_to_one"
    )

    if (
        len(merged) != len(model_a)
        or len(merged) != len(model_b)
    ):
        raise ValueError(
            "Modellerin gozlem sayilari farkli."
        )

    if not (
        merged["y_a"].to_numpy()
        == merged["y_b"].to_numpy()
    ).all():
        raise ValueError(
            "Modellerin hedef degerleri farkli."
        )

    if not (
        merged["zape_a"].to_numpy()
        == merged["zape_b"].to_numpy()
    ).all():
        raise ValueError(
            "Modellerin ZAPE bilgileri farkli."
        )

    y = merged["y_a"].to_numpy()

    p_a = merged["p_a"].to_numpy()
    p_b = merged["p_b"].to_numpy()

    zape = merged["zape_a"].to_numpy()

    metrics_a = calculate_metrics(y, p_a)
    metrics_b = calculate_metrics(y, p_b)

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

        sample_a = calculate_metrics(
            y_sample,
            p_a[sampled_indices]
        )

        sample_b = calculate_metrics(
            y_sample,
            p_b[sampled_indices]
        )

        for metric in differences:
            differences[metric].append(
                sample_b[metric] - sample_a[metric]
            )

    rows = []

    for metric, values in differences.items():

        values = np.asarray(values)

        if len(values) == 0:
            raise ValueError(
                "Gecerli bootstrap ornegi yok."
            )

        lower, upper = np.percentile(
            values,
            [2.5, 97.5]
        )

        rows.append({
            "Senaryo": scenario,
            "Model_A": name_a,
            "Model_B": name_b,
            "Metric": metric,
            "Score_A": metrics_a[metric],
            "Score_B": metrics_b[metric],
            "Difference": (
                metrics_b[metric] - metrics_a[metric]
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
    print("GOREV 22.5 - MODEL BOOTSTRAP")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Sonuc dosyasi zaten mevcut: {OUTPUT_PATH}"
        )

    for model_name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name}: {path}"
            )

    rng = np.random.default_rng(RANDOM_STATE)

    all_results = []

    for scenario in SCENARIOS:

        print("\n" + "=" * 60)
        print("SENARYO:", scenario)
        print("=" * 60)

        predictions = {
            name: load_model_predictions(path, scenario)
            for name, path in MODEL_FILES.items()
        }

        for name_a, name_b in combinations(
            MODEL_FILES.keys(),
            2
        ):

            rows = compare_models(
                predictions[name_a],
                predictions[name_b],
                name_a,
                name_b,
                scenario,
                rng
            )

            all_results.extend(rows)

            print(f"\n{name_a} -> {name_b}")

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

    print("\nSonuc dosyasi:", OUTPUT_PATH)
    print("\nGOREV 22.5 TAMAMLANDI")


if __name__ == "__main__":
    main()