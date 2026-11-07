from src.config import DATA_PATH as CONFIG_DATA_PATH
import pandas as pd
import numpy as np

DATA_PATH = str(CONFIG_DATA_PATH)

df = pd.read_csv(DATA_PATH)

# Tarih bilgilerini düzenle
df["_tarih"] = pd.to_datetime(
    df["DATA AMOSTRAGEM"],
    errors="coerce"
)

# Orijinal satır sırasını koru
df["_orijinal_sira"] = np.arange(len(df))

# FREGUESIA bazında tarih sırasına göre sırala
df = df.sort_values(
    ["FREGUESIA", "_tarih", "_orijinal_sira"],
    kind="stable"
).copy()

tests = [
    ("Yagis LAG1", "PRECIPITAÇÃO (mm)", "PRECIP_LAG1_REGIONAL", "lag"),
    ("Sicaklik LAG1", "TEMP MÁX (ºC)", "TEMP_LAG1_REGIONAL", "lag"),
    ("Yagis EWMA", "PRECIPITAÇÃO (mm)", "PRECIP_EWMA_REGIONAL", "ewma"),
    ("Sicaklik EWMA", "TEMP MÁX (ºC)", "TEMP_EWMA_REGIONAL", "ewma"),
]

print("\n========== FREGUESIA BAZLI ZAMAN OZELLIKLERI DENETIMI ==========")

for name, source, target, method in tests:

    if method == "lag":
        expected = df.groupby("FREGUESIA")[source].shift(1)
    else:
        expected = df.groupby("FREGUESIA")[source].transform(
            lambda x: x.ewm(alpha=0.5, adjust=True).mean()
        )

    actual = df[target]

    matches = np.isclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-5,
        atol=1e-6
    )

    print(f"\n{name}")
    print("Eslesen:", int(matches.sum()))
    print("Toplam:", len(matches))
    print("Uyumsuz:", int((~matches).sum()))

    if (~matches).any():
        example = df.loc[
            ~matches,
            ["FREGUESIA", "DATA AMOSTRAGEM", source, target]
        ].head(5)

        print("\nIlk 5 uyumsuz gozlem:")
        print(example.to_string(index=False))

print("\nKontrol tamamlandi.")