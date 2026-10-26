import numpy as np
import pandas as pd

DATA_PATH = "data/raw/Base_Tech4Aqua_Features_Avancadas.csv"

columns = [
    "ZAPE NOME",
    "DATA AMOSTRAGEM",
    "Desinfetante residual",
    "DELTA_DIAS_ZAPE",
    "LOG_DELTA_DIAS",
    "CLORO_RESIDUAL_PREV",
]

df = pd.read_csv(DATA_PATH, usecols=columns).copy()

df["_tarih"] = pd.to_datetime(
    df["DATA AMOSTRAGEM"],
    errors="coerce"
)

df["_orijinal_sira"] = np.arange(len(df))

df = df.sort_values(
    ["ZAPE NOME", "_tarih", "_orijinal_sira"],
    kind="stable"
).copy()

# 1. Önceki ZAPE ölçümüyle gün farkı
expected_delta = (
    df.groupby("ZAPE NOME")["_tarih"]
    .diff()
    .dt.days
)

# 2. Gün farkının logaritması
expected_log = np.log1p(df["DELTA_DIAS_ZAPE"])

# 3. Önceki klor ölçümü
expected_chlorine = (
    df.groupby("ZAPE NOME")["Desinfetante residual"]
    .shift(1)
)

tests = [
    (
        "DELTA_DIAS_ZAPE",
        df["DELTA_DIAS_ZAPE"],
        expected_delta,
    ),
    (
        "LOG_DELTA_DIAS",
        df["LOG_DELTA_DIAS"],
        expected_log,
    ),
    (
        "CLORO_RESIDUAL_PREV",
        df["CLORO_RESIDUAL_PREV"],
        expected_chlorine,
    ),
]

print("\n========== ZAPE GECMIS OZELLIKLERI DENETIMI ==========")

for name, actual, expected in tests:

    matches = np.isclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        equal_nan=True,
        rtol=1e-5,
        atol=1e-6,
    )

    print(f"\n{name}")
    print("Eslesen:", int(matches.sum()))
    print("Toplam:", len(matches))
    print("Uyumsuz:", int((~matches).sum()))

    if (~matches).any():

        example = df.loc[
            ~matches,
            [
                "ZAPE NOME",
                "DATA AMOSTRAGEM",
                "Desinfetante residual",
                "DELTA_DIAS_ZAPE",
                "LOG_DELTA_DIAS",
                "CLORO_RESIDUAL_PREV",
            ],
        ].head(5)

        print("\nIlk 5 uyumsuz gozlem:")
        print(example.to_string(index=False))

print("\nKontrol tamamlandi.")