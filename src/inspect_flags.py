
import pandas as pd

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

df = pd.read_csv(DATA_PATH)

# Modelin kullanildigi etiketli gozlemler
df = df.dropna(subset=["Target_Coliformes"]).copy()

pairs = [
    ("FLAG_CLORO_CRITICO", "Desinfetante residual"),
    ("FLAG_TURVACAO_ALTA", "Turvação"),
    ("FLAG_PH_ANORMAL", "pH"),
    ("FLAG_CALOR_EXTREMO", "TEMP MÁX (ºC)"),
    ("FLAG_VERAO", "MES"),
]

print("FLAG - OLCUM ILISKISI KONTROLU")
print("=" * 60)

for flag, measurement in pairs:
    print(f"\nFLAG: {flag}")
    print(f"OLCUM: {measurement}")

    if flag not in df.columns or measurement not in df.columns:
        print("Gerekli sutun bulunamadi.")
        continue

    subset = df[[flag, measurement]].copy()

    print("\nFLAG deger sayilari:")
    print(subset[flag].value_counts(dropna=False).to_string())

    print("\nOlcumun FLAG'a gore dagilimi:")
    print(
        subset.groupby(flag)[measurement]
        .agg(["count", "min", "median", "max"])
        .round(4)
        .to_string()
    )

    print("\nIlk 5 ornek:")
    print(subset.dropna().head(5).to_string(index=False))

print("\nKontrol tamamlandi.")
