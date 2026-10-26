
from pathlib import Path
import pandas as pd

from src.preprocessing import load_and_preprocess_data
from src.cold_start_features import RESTRICTED_FEATURES

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

X, y, groups, feature_names = load_and_preprocess_data(DATA_PATH)

# Ilk inceleme icin ozellikleri siniflandiriyoruz.
# Bunlar kesin karar degil, inceleme oncelikleridir.

# Gecmis olcum veya zaman farki kullanimini
# gosterebilecek daha belirgin isim kaliplari
history_keywords = [
    "_PREV",
    "_LAG",
    "_EWMA",
    "_DELTA",
    "DELTA_DIAS",
]

derived_keywords = [
    "FLAG_",
    "INDICE_",
    "PROXY_",
    "DEGRADACAO_",
    "_DIFF",
    "SIN_",
    "COS_",
    "LOG_",
]

derived_keywords = [
    "FLAG", "INDICE", "PROXY", "DEGRADACAO",
    "DIFF", "SIN_", "COS_", "LOG_"
]

rows = []

for col in X.columns:
    upper = col.upper()

    if col in RESTRICTED_FEATURES:
        category = "Kisitli senaryo"
    elif any(k in upper for k in history_keywords):
        category = "Gecmis veri kontrolu"
    elif any(k in upper for k in derived_keywords):
        category = "Formul kontrolu"
    else:
        category = "Kaynak kontrolu"

    rows.append({
        "Ozellik": col,
        "Kategori": category,
        "Eksik_Sayi": int(X[col].isna().sum()),
        "Eksik_Oran": round(
            float(X[col].isna().mean()), 4
        ),
        "Farkli_Deger": int(X[col].nunique()),
        "Kaynak_Dogrulandi": "Hayir",
        "Yeni_Bolgede_Erisilebilir": "Bilinmiyor",
        "Not": "",
    })

audit = pd.DataFrame(rows)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "feature_provenance_audit.csv"
audit.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nOZELLIK KAYNAGI DENETIMI")
print("=" * 70)
print(audit[
    ["Ozellik", "Kategori", "Eksik_Oran"]
].to_string(index=False))

print("\nKategori dagilimi:")
print(audit["Kategori"].value_counts().to_string())

print(f"\nDosya kaydedildi: {output_path}")
