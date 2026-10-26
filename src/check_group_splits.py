
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import load_and_preprocess_data

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

X, y, groups, feature_names = load_and_preprocess_data(
    DATA_PATH
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("ZAPE BAZLI VERI AYRIMI KONTROLU")
print("=" * 65)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    common_groups = train_groups.intersection(test_groups)

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    result = {
        "Fold": fold,
        "Egitim_Gozlem": len(train_idx),
        "Test_Gozlem": len(test_idx),
        "Egitim_ZAPE": len(train_groups),
        "Test_ZAPE": len(test_groups),
        "Ortak_ZAPE": len(common_groups),
        "Egitim_Pozitif": int(y_train.sum()),
        "Test_Pozitif": int(y_test.sum()),
        "Test_Pozitif_Oran": round(
            float(y_test.mean()), 4
        ),
    }

    results.append(result)

    print(
        f"Fold {fold}: "
        f"Egitim ZAPE={len(train_groups)}, "
        f"Test ZAPE={len(test_groups)}, "
        f"Ortak ZAPE={len(common_groups)}, "
        f"Test pozitif={int(y_test.sum())}/{len(y_test)}"
    )

    if common_groups:
        raise ValueError(
            f"Fold {fold}: Bolge sizintisi tespit edildi!"
        )

results_df = pd.DataFrame(results)

print("\nDETAYLI TABLO")
print(results_df.to_string(index=False))

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "group_split_audit.csv",
    index=False
)

print("\nKontrol tamamlandi.")
print("Bolge cakismasi tespit edilmedi.")
