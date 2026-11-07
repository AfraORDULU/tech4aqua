from src.config import DATA_PATH as CONFIG_DATA_PATH

from pathlib import Path
import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data

# Projenin ana klasöründeki veri dosyası
DATA_PATH = Path(str(CONFIG_DATA_PATH))

print("=" * 60)
print("TECH4AQUA - COLD START OZELLIK DENETIMI")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Veri dosyasi bulunamadi: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\n1. VERI BOYUTU")
print("Satir:", len(df))
print("Sutun:", len(df.columns))

print("\n2. HEDEF DAGILIMI")
target = df["Target_Coliformes"].dropna().astype(int)
print(target.value_counts().sort_index())
print("Etiketli gozlem:", len(target))

print("\n3. MODELE GIREN OZELLIKLER")
try:
    X, y, groups, feature_groups = load_and_preprocess_data(
        str(DATA_PATH)
    )

    for i, col in enumerate(X.columns, start=1):
        print(f"{i:02d}. {col}")

    print("\nToplam ozellik:", X.shape[1])
    print("Toplam grup:", groups.nunique())

    print("\n4. COLD-START ICIN SUPHELI OZELLIKLER")

    suspicious_terms = [
        "MICROBIO",
        "COLIFORM",
        "BACT",
        "FALHA",
        "STREAK",
        "TARGET",
        "ALVO",
        "Y_BIN",
    ]

    suspicious = [
        col for col in X.columns
        if any(term in col.upper() for term in suspicious_terms)
    ]

    if suspicious:
        for col in suspicious:
            print("KONTROL GEREKIYOR:", col)
    else:
        print("Isim taramasinda supheli sutun bulunamadi.")

    print("\n5. OZELLIK GRUPLARI")
    for name, cols in feature_groups.items():
        print(f"\n{name}: {len(cols)} ozellik")
        for col in cols:
            print(" -", col)

except Exception as exc:
    print("\nON ISLEME HATASI:")
    print(type(exc).__name__, str(exc))
    raise

print("\nDenetim tamamlandi.")
