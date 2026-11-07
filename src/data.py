"""
Tech4Aqua - Modelleme verisini yukler ve ozellik setlerini tanimlar.

Butun deneyler veriyi bu dosyadan alir. Boylece her deney ayni
satirlari, ayni etiketi ve ayni ozellik listelerini kullanir.

Donen tabloda iki tur sutun vardir:
  * Ozellik sutunlari: FEATURE_SETS icindeki adlar. Modele yalnizca
    bunlar girer.
  * Analiz sutunlari (kucuk harfli): y, zape, date, municipality,
    year, is_investigation, chlorine_missing. Bunlar MODELE GIRMEZ;
    yalnizca bolme, filtreleme ve raporlama icindir.
"""

import pandas as pd

from src.config import DATA_PATH

TARGET_RAW = "Bactérias coliformes"
ZAPE_COL = "ZAPE NOME"
DATE_COL = "DATA AMOSTRAGEM"
CHLORINE = "Desinfetante residual"

# Ara rapordaki uc ozellik seti + yalniz klor.
FEATURE_SETS = {
    "A": [  # klor + hava + ay
        CHLORINE,
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "MES",
    ],
    "B": [  # yalniz hava + takvim
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "MES",
        "SIN_MES",
        "COS_MES",
        "FLAG_VERAO",
    ],
    "C": [  # A + alan bilgisine dayali bayraklar
        CHLORINE,
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "FLAG_CLORO_CRITICO",
        "FLAG_CALOR_EXTREMO",
        "FLAG_RUNOFF_RISCO",
        "DEGRADACAO_TERMICA_CLORO",
        "MES",
        "SIN_MES",
        "COS_MES",
        "FLAG_VERAO",
    ],
    "CHLORINE_ONLY": [CHLORINE],
}

ANALYSIS_COLS = [
    "y", "zape", "date", "municipality", "year",
    "is_investigation", "chlorine_missing",
]

# Adinda bunlardan biri gecen sutun asla ozellik olamaz
# (hedefin kendisi, ayni numunedeki bakteri olcumleri, gecmis etiketler).
FORBIDDEN_KEYWORDS = ("coli", "bact", "target", "falha", "microbio", "meta_")


def check_features(columns):
    """Ozellik listesinde sizinti riski tasiyan sutun varsa hata verir."""
    bad = [
        c for c in columns
        if any(k in c.lower() for k in FORBIDDEN_KEYWORDS)
        or c in ANALYSIS_COLS
    ]
    if bad:
        raise ValueError(f"Yasakli ozellik sutunlari: {bad}")


def load_dataset(path=DATA_PATH):
    """Koliform sonucu olan 522 numuneyi analiz sutunlariyla dondurur."""
    raw = pd.read_csv(path)
    if "META_IS_INVESTIGATION" not in raw.columns:
        raise RuntimeError(
            "Islenmis veri bulunamadi. Once calistirin: "
            "python -m src.build_dataset"
        )

    df = raw[raw[TARGET_RAW].notna()].reset_index(drop=True).copy()

    df["y"] = (df[TARGET_RAW] > 0).astype(int)
    df["zape"] = df[ZAPE_COL].astype(str)
    df["date"] = pd.to_datetime(df[DATE_COL])
    df["municipality"] = df["META_MUNICIPALITY"].astype(str)
    df["year"] = df["date"].dt.year
    df["is_investigation"] = df["META_IS_INVESTIGATION"].astype(bool)
    df["chlorine_missing"] = df[CHLORINE].isna()

    for name, cols in FEATURE_SETS.items():
        check_features(cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} setinde eksik sutunlar: {missing}")

    return df
