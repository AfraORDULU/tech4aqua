
from pathlib import Path
import pandas as pd

path = Path(
    "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"
)

df = pd.read_csv(path)

# Yalnizca hedef etiketi bulunan gozlemler
df = df.dropna(subset=["Target_Coliformes"]).copy()

columns = [
    "CLORO_RESIDUAL_PREV",
    "DELTA_DIAS_ZAPE",
    "LOG_DELTA_DIAS",
    "Proxy_Tempo_Retencao",
    "INDICE_ESTRESSE_CONJUNTO",
]

print("COLD-START OZELLIK KONTROLU")
print("=" * 50)

for col in columns:
    print(f"\nOZELLIK: {col}")

    if col not in df.columns:
        print("Sutun bulunamadi.")
        continue

    print("Eksik deger:", df[col].isna().sum())
    print("Farkli deger:", df[col].nunique())
    print("Minimum:", df[col].min())
    print("Maksimum:", df[col].max())

print("\nKontrol tamamlandi.")
