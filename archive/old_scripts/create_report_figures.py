from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


TABLE_DIR = Path("results/report_tables")
FIGURE_DIR = Path("results/report_figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300
})


# ==================================================
# 1. MODEL KARSILASTIRMA GRAFIKLERI
# ==================================================

df = pd.read_csv(
    TABLE_DIR / "table_01_model_comparison.csv"
)

feature_sets = [
    "A_Conservative",
    "B_Meteorological",
    "C_Extended"
]

models = [
    "Random Forest",
    "LightGBM",
    "Logistic Regression"
]

metric_info = [
    ("ROC_AUC", "ROC-AUC", "figure_01_roc_auc.png"),
    ("PR_AUC", "PR-AUC", "figure_02_pr_auc.png"),
    ("MCC", "MCC", "figure_03_mcc.png")
]

for metric, metric_label, filename in metric_info:

    pivot = df.pivot(
        index="Feature_Set",
        columns="Model",
        values=metric
    )

    pivot = pivot.loc[feature_sets, models]

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.75
    )

    ax.set_title(
        f"Model Performance Comparison - {metric_label}"
    )

    ax.set_xlabel("Feature Set")
    ax.set_ylabel(metric_label)

    ax.set_xticklabels(
        ["A - Conservative", "B - Meteorological", "C - Extended"],
        rotation=0
    )

    ax.legend(
        title="Model",
        loc="upper right"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            fontsize=8
        )

    ax.set_ylim(
        0,
        max(0.1, df[metric].max() * 1.20)
    )

    plt.tight_layout()

    output = FIGURE_DIR / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Olusturuldu:", output)


# ==================================================
# 2. ABLASYON GRAFIGI
# ==================================================

ablation = pd.read_csv(
    TABLE_DIR / "table_03_ablation.csv"
)

ablation = ablation.set_index("Scenario")

scenarios = [
    "C_Extended",
    "C_NoThermalChlorine"
]

metrics = [
    "ROC_AUC",
    "PR_AUC",
    "MCC"
]

means = np.array([
    [
        ablation.loc[scenario, f"{metric}_mean"]
        for metric in metrics
    ]
    for scenario in scenarios
])

stds = np.array([
    [
        ablation.loc[scenario, f"{metric}_std"]
        for metric in metrics
    ]
    for scenario in scenarios
])

x = np.arange(len(metrics))

width = 0.35

fig, ax = plt.subplots(
    figsize=(9, 6)
)

for i, scenario in enumerate(scenarios):

    positions = x + (i - 0.5) * width

    bars = ax.bar(
        positions,
        means[i],
        width,
        yerr=stds[i],
        capsize=4,
        label=scenario
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=5,
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.set_ylabel("Metric Value")

ax.set_title(
    "LightGBM - Thermal Chlorine Feature Ablation"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_axisbelow(True)

ax.set_ylim(
    0,
    max(means.flatten()) * 1.25
)

plt.tight_layout()

output = FIGURE_DIR / "figure_04_ablation.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Olusturuldu:", output)

print("\n" + "=" * 65)
print("GOREV 25.4 - AKADEMIK GRAFIKLER TAMAMLANDI")
print("=" * 65)

print("Grafik klasoru:", FIGURE_DIR)
print("Toplam grafik:", len(list(FIGURE_DIR.glob("figure_*.png"))))