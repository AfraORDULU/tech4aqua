import pandas as pd
import numpy as np

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

columns = [
    "FREGUESIA",
    "DATA AMOSTRAGEM",
    "PRECIPITAÇÃO (mm)",
    "TEMP MÁX (ºC)",
    "PRECIP_LAG1_REGIONAL",
    "TEMP_LAG1_REGIONAL",
    "PRECIP_EWMA_REGIONAL",
    "TEMP_EWMA_REGIONAL",
]

df = pd.read_csv(DATA_PATH, usecols=columns).copy()

df["_tarih"] = pd.to_datetime(
    df["DATA AMOSTRAGEM"],
    errors="coerce"
)

df["_orijinal_sira"] = np.arange(len(df))

# Orijinal sıralama
original = df.sort_values(
    ["FREGUESIA", "_tarih", "_orijinal_sira"],
    kind="stable"
).copy()

# Aynı gün içindeki kayıt sırasını tersine çevir
reversed_order = df.sort_values(
    ["FREGUESIA", "_tarih", "_orijinal_sira"],
    ascending=[True, True, False],
    kind="stable"
).copy()

tests = [
    ("Yagis LAG1", "PRECIPITAÇÃO (mm)", "PRECIP_LAG1_REGIONAL", "lag"),
    ("Sicaklik LAG1", "TEMP MÁX (ºC)", "TEMP_LAG1_REGIONAL", "lag"),
    ("Yagis EWMA", "PRECIPITAÇÃO (mm)", "PRECIP_EWMA_REGIONAL", "ewma"),
    ("Sicaklik EWMA", "TEMP MÁX (ºC)", "TEMP_EWMA_REGIONAL", "ewma"),
]

print("\n========== AYNI GUN SIRALAMA ETKISI ==========")

for name, source, target, method in tests:

    if method == "lag":
        recalculated = reversed_order.groupby(
            "FREGUESIA"
        )[source].shift(1)

    else:
        recalculated = reversed_order.groupby(
            "FREGUESIA"
        )[source].transform(
            lambda x: x.ewm(alpha=0.5, adjust=True).mean()
        )

    # Satır kimliğine göre orijinal sıraya dön
    recalculated.index = reversed_order["_orijinal_sira"]
    recalculated = recalculated.reindex(
        original["_orijinal_sira"]
    )

    actual = original[target].to_numpy(dtype=float)

    matches = np.isclose(
        actual,
        recalculated.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-5,
        atol=1e-6
    )

    print(f"\n{name}")
    print("Degisen satir:", int((~matches).sum()))
    print("Toplam satir:", len(matches))
    print("Degisim orani:", f"%{(~matches).mean() * 100:.2f}")

print("\nKontrol tamamlandi.")