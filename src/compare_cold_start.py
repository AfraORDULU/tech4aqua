
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    precision_score,
)

from src.preprocessing import load_and_preprocess_data
from src.cold_start_features import (
    build_cold_start_sets,
    build_conservative_set,
)

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

# Mevcut veri hazirlama fonksiyonumuzu kullaniyoruz
X, y, groups, feature_names = load_and_preprocess_data(DATA_PATH)

X_basic, X_extended = build_cold_start_sets(X)
X_conservative = build_conservative_set(X)

# Iki senaryoda da ayni bolgeler test edilecek
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

splits = list(cv.split(X, y, groups))

results = []

for scenario_name, X_scenario in [
    ("Ihtiyatli", X_conservative),
    ("Temel", X_basic),
    ("Genisletilmis", X_extended),
]:
    print(f"\nSenaryo: {scenario_name}")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X_scenario.iloc[train_idx]
        X_test = X_scenario.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # ROC-AUC ve PR-AUC icin testte iki sinif gerekir
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"Fold {fold}: Tek sinif nedeniyle atlandi.")
            continue

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        
        

        result = {
            "Senaryo": scenario_name,
            "Fold": fold,
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "PR_AUC": average_precision_score(
                y_test, probabilities
            ),
            "MCC": matthews_corrcoef(y_test, predictions),
            "Test_Sayisi": len(y_test),
            "Pozitif_Sayisi": int(y_test.sum()),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Recall": float(recall),
            "Precision": float(precision),

        }

        results.append(result)

        print(
            f"Fold {fold} | "
            f"ROC-AUC: {result['ROC_AUC']:.4f} | "
            f"PR-AUC: {result['PR_AUC']:.4f} | "
            f"MCC: {result['MCC']:.4f}"
        )


results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("Gecerli fold sonucu olusmadi.")

print("\n========== ORTALAMA SONUCLAR ==========")


summary = results_df.groupby("Senaryo")[
    ["ROC_AUC", "PR_AUC", "MCC", "Recall", "Precision"]
].agg(["mean", "std"])

print("\n========== TOPLAM HATA SAYILARI ==========")

error_summary = results_df.groupby("Senaryo")[
    ["TP", "TN", "FP", "FN"]
].sum()

print(error_summary.to_string())


print(summary.round(4).to_string())

# Sonuclari kaydediyoruz
from pathlib import Path

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "cold_start_comparison.csv",
    index=False
)

summary.to_csv(
    output_dir / "cold_start_summary.csv"
)

print("\nSonuclar results/ klasorune kaydedildi.")
