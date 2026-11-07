from src.config import DATA_PATH as CONFIG_DATA_PATH

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing import load_and_preprocess_data
from src.cold_start_features import (
    build_cold_start_sets,
    build_conservative_set,
)

DATA_PATH = str(CONFIG_DATA_PATH)

# 1. Veriyi hazirla
X, y, groups, _ = load_and_preprocess_data(DATA_PATH)

X_basic, X_extended = build_cold_start_sets(X)
X_conservative = build_conservative_set(X)

scenarios = [
    ("Ihtiyatli", X_conservative),
    ("Temel", X_basic),
    ("Genisletilmis", X_extended),
]

thresholds = [0.20, 0.30, 0.40, 0.50]

# 2. Ayni ZAPE bazli veri bolmelerini kullan
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

splits = list(cv.split(X, y, groups))

results = []

for scenario_name, X_scenario in scenarios:

    print(f"\nSENARYO: {scenario_name}")
    print("=" * 60)

    # Her gozlemin test fold'undaki tahmini
    probabilities = np.full(len(X_scenario), np.nan)
    model_predictions = np.full(len(X_scenario), -1, dtype=int)

    # 3. Modelleri egit
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):

        X_train = X_scenario.iloc[train_idx]
        X_test = X_scenario.iloc[test_idx]
        y_train = y.iloc[train_idx]

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

        probabilities[test_idx] = model.predict_proba(X_test)[:, 1]
        model_predictions[test_idx] = model.predict(X_test)

    # 4. Tum gozlemler tahmin edilmis mi?
    if np.isnan(probabilities).any():
        raise ValueError("Bazi gozlemler icin olasilik eksik!")

    if (model_predictions == -1).any():
        raise ValueError("Bazi gozlemler icin sinif tahmini eksik!")

    y_true = y.to_numpy()

    # 5. Tam 0.50 olasilikli gozlemleri say
    exact_half_count = int(np.sum(probabilities == 0.50))

    print(
        f"Tam 0.50 olasilikli gozlem sayisi: "
        f"{exact_half_count}"
    )

    # 6. model.predict() ile dogrudan kontrol
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        model_predictions,
        labels=[0, 1],
    ).ravel()

    print("\nMODEL.PREDICT() SONUCLARI")
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    # 7. Farkli karar esiklerini incele
    for threshold in thresholds:

        # Tam esikteki degerleri pozitif saymiyoruz
        predicted = (probabilities > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted,
            labels=[0, 1],
        ).ravel()

        results.append({
            "Senaryo": scenario_name,
            "Esik": threshold,
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "Recall": recall_score(
                y_true,
                predicted,
                zero_division=0,
            ),
            "Precision": precision_score(
                y_true,
                predicted,
                zero_division=0,
            ),
            "F1": f1_score(
                y_true,
                predicted,
                zero_division=0,
            ),
        })

        # 0.50 esigini model.predict() ile karsilastir
        if threshold == 0.50:
            difference_count = int(
                np.sum(predicted != model_predictions)
            )

            print(
                "\n0.50 ESIGI - MODEL.PREDICT() FARKI: "
                f"{difference_count} gozlem"
            )

# 8. Sonuc tablosu
results_df = pd.DataFrame(results)

print("\n========== ESIK KARSILASTIRMASI ==========")
print(results_df.round(4).to_string(index=False))

# 9. CSV dosyasina kaydet
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "threshold_analysis.csv",
    index=False,
)

print(
    "\nSonuclar results/threshold_analysis.csv "
    "dosyasina kaydedildi."
)
