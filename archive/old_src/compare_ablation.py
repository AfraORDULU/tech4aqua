from src.config import DATA_PATH as CONFIG_DATA_PATH

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing import load_and_preprocess_data
from src.cold_start_ablation import build_ablation_sets


# 1. Veri setini oku
DATA_PATH = str(CONFIG_DATA_PATH)

X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

scenarios = build_ablation_sets(X)

# 2. Ayni bolge bazli veri bolmelerini olustur
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

splits = list(cv.split(X, y, groups))

results = []

# 3. Her ozellik senaryosunu test et
for scenario_name, X_scenario in scenarios.items():

    print(f"\nSENARYO: {scenario_name}")
    print("-" * 65)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X_scenario.iloc[train_idx]
        X_test = X_scenario.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Her fold icin model bastan egitilir
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        # Siniflandirma sonuclari
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        # Performans metrikleri
        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc = average_precision_score(y_test, probabilities)
        mcc = matthews_corrcoef(y_test, predictions)

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Senaryo": scenario_name,
            "Fold": fold,
            "Ozellik_Sayisi": X_scenario.shape[1],
            "Test_N": len(y_test),
            "Test_Pozitif": int(y_test.sum()),
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "MCC": mcc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
        })

        print(
            f"Fold {fold} | "
            f"ROC-AUC: {roc_auc:.4f} | "
            f"PR-AUC: {pr_auc:.4f} | "
            f"MCC: {mcc:.4f} | "
            f"Recall: {recall:.4f}"
        )

# 4. Ozet tablosu
results_df = pd.DataFrame(results)

summary = results_df.groupby("Senaryo").agg(
    Ozellik_Sayisi=("Ozellik_Sayisi", "first"),
    ROC_AUC=("ROC_AUC", "mean"),
    PR_AUC=("PR_AUC", "mean"),
    MCC=("MCC", "mean"),
    Recall=("Recall", "mean"),
    Precision=("Precision", "mean"),
    F1=("F1", "mean"),
    TP=("TP", "sum"),
    FN=("FN", "sum"),
    FP=("FP", "sum"),
    TN=("TN", "sum"),
)

summary = summary.sort_values(
    by="PR_AUC",
    ascending=False
)

print("\n========== ABLASYON OZETI ==========")
print(summary.round(4).to_string())

# 5. Sonuclari kaydet
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "ablation_results.csv",
    index=False,
)

summary.to_csv(
    output_dir / "ablation_summary.csv"
)

print("\nSonuclar results/ klasorune kaydedildi.")
