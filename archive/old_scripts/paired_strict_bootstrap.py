from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


# ==================================================
# 1. DOSYA YOLLARI VE AYARLAR
# ==================================================

RESULTS_DIR = Path("results")

# Önceki deney: Meteorolojik geçmiş özellikleri çıkarılmış modeller
OLD_PATH = RESULTS_DIR / "temporal_ablation_oof.csv"

# Yeni deney: ZAPE geçmiş özellikleri de çıkarılmış modeller
NEW_PATH = RESULTS_DIR / "strict_cold_start_oof.csv"

# Sonuçların kaydedileceği yeni dosya
OUTPUT_PATH = RESULTS_DIR / "paired_strict_bootstrap.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

# Karşılaştırılacak modeller
COMPARISONS = [
    ("F_NoTemporal", "F_StrictColdStart"),
    ("G_NoTemporal", "G_StrictColdStart"),
]


# ==================================================
# 2. OOF TAHMİNLERİNİ HAZIRLAMA
# ==================================================

def prepare_predictions(df, scenario_name):

    subset = df.loc[
        df["Senaryo"] == scenario_name
    ].copy()

    if subset.empty:
        raise ValueError(
            f"Senaryo bulunamadi: {scenario_name}"
        )

    # Her gözlem için 10 seed tahmininin ortalamasını al
    grouped = (
        subset.groupby("idx", as_index=False)
        .agg(
            zape=("zape", "first"),
            y=("y", "first"),
            p=("p", "mean"),
            seed_count=("Seed", "nunique"),
        )
    )

    # Her gözlem için 10 seed bulunduğunu kontrol et
    if not (grouped["seed_count"] == 10).all():
        raise ValueError(
            f"{scenario_name}: Her gozlem icin 10 seed bulunmuyor."
        )

    return grouped[["idx", "zape", "y", "p"]]


# ==================================================
# 3. PERFORMANS METRİKLERİNİ HESAPLAMA
# ==================================================

def calculate_metrics(y, probabilities):

    return {
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(y, probabilities),
    }


# ==================================================
# 4. EŞLEŞTİRİLMİŞ ZAPE BOOTSTRAP
# ==================================================

def bootstrap_comparison(base_df, reduced_df, rng):

    # Aynı gözlemleri eşleştir
    merged = base_df.merge(
        reduced_df,
        on="idx",
        suffixes=("_base", "_reduced"),
        validate="one_to_one",
    )

    # Hedef değerlerinin aynı olduğunu kontrol et
    if not (
        merged["y_base"].to_numpy()
        == merged["y_reduced"].to_numpy()
    ).all():
        raise ValueError(
            "Hedef degerleri uyusmuyor."
        )

    # ZAPE gruplarının aynı olduğunu kontrol et
    if not (
        merged["zape_base"].to_numpy()
        == merged["zape_reduced"].to_numpy()
    ).all():
        raise ValueError(
            "ZAPE gruplari uyusmuyor."
        )

    y = merged["y_base"].to_numpy()

    p_base = merged["p_base"].to_numpy()
    p_reduced = merged["p_reduced"].to_numpy()

    # Orijinal veri üzerindeki performanslar
    original_base = calculate_metrics(
        y,
        p_base
    )

    original_reduced = calculate_metrics(
        y,
        p_reduced
    )

    # Performans farkı: Strict - NoTemporal
    original_diff = {
        metric: (
            original_reduced[metric]
            - original_base[metric]
        )
        for metric in ["ROC_AUC", "PR_AUC"]
    }

    # ZAPE bazında bootstrap hazırlığı
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

    # ==================================================
    # 5. BOOTSTRAP ÖRNEKLEMESİ
    # ==================================================

    for _ in range(N_BOOTSTRAP):

        # ZAPE gruplarını tekrar seç
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        # Seçilen ZAPE'lerin tüm gözlemlerini al
        sampled_indices = np.concatenate([
            group_indices[group]
            for group in sampled_groups
        ])

        sampled_y = y[sampled_indices]

        # ROC-AUC için iki sınıfın da bulunması gerekir
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

        # Her metrik için performans farkını kaydet
        for metric in bootstrap_diffs:

            difference = (
                reduced_metrics[metric]
                - base_metrics[metric]
            )

            bootstrap_diffs[metric].append(
                difference
            )

    # ==================================================
    # 6. %95 GÜVEN ARALIKLARI
    # ==================================================

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


# ==================================================
# 7. ANA PROGRAM
# ==================================================

def main():

    # Önceki ve yeni deney tahminlerini oku
    old_df = pd.read_csv(OLD_PATH)
    new_df = pd.read_csv(NEW_PATH)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    all_results = []

    print(
        "\n========== STRICT COLD-START BOOTSTRAP =========="
    )

    # İki model çiftini karşılaştır
    for base_name, reduced_name in COMPARISONS:

        print(
            f"\n{base_name} -> {reduced_name}"
        )

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

    # ==================================================
    # 8. SONUÇLARI KAYDET
    # ==================================================

    result_df = pd.DataFrame(
        all_results
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSonuclar kaydedildi: {OUTPUT_PATH}"
    )

    print(
        "\nAnaliz tamamlandi."
    )


if __name__ == "__main__":
    main()