"""
Tech4Aqua - Islenmis (etiketi kontrol edilmis) veri setini uretir.

Neden gerekli?
--------------
Mendeley paketindeki Base_Tech4Aqua_Features_Avancadas.csv, yazarlarin
01_feature_engineering.py betigiyle uretilmis. Bu betik sonucu sayiya
cevirirken '<' ve '>' isaretlerini siliyor. Bu nedenle '<0.1'
("tespit siniri altinda", yani bakteri bulunmadi) sonucu 0.1'e donusup
POZITIF etiketleniyor. Ayrica ozellik tablosunda numunenin
NEDEN alindigi (rutin izleme mi, bir uygunsuzluk sonrasi sorusturma mi)
bilgisi kayboluyor.

Bu betik:
  1. Ham kayitlardan (Base_Revisada_Tech4Aqua.xlsx) koliform sonuclarini
     operator bilgisiyle birlikte yeniden okur.
  2. '<' operatorlu sonuclari 0 (tespit edilmedi) kabul eder.
  3. Ayni ZAPE + ayni gun icin birden fazla kayit varsa birlestirir
     (herhangi biri pozitifse numune pozitif).
  4. Ornekleme turu, laboratuvar ve kaynak dosya bilgisini META_
     sutunlari olarak ekler. META_ sutunlari ASLA modele girmez
     (src/preprocessing.py bunlari yasaklar); yalnizca analiz ve
     duyarlilik deneyleri icindir.
  5. Sonucu data/processed/tech4aqua_labeled.csv dosyasina yazar ve
     degisen etiketleri results/data_audit/label_changes.csv'ye kaydeder.

Calistirma (proje kok klasorunden):
    python -m src.build_dataset
"""

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_CSV,
    RAW_XLSX,
    PROCESSED_CSV,
    PROCESSED_DIR,
    DATA_AUDIT_DIR,
)

TARGET_COL = "Bactérias coliformes"
KEYS = ["ZAPE NOME", "DATE_KEY"]
INVESTIGATION_TEXT = "Investiga"


def parse_result(value):
    """'0,2' / '300.00000000' gibi metinleri sayiya cevirir."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", ".").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_raw_coliform_records():
    raw = pd.read_excel(RAW_XLSX, sheet_name="MEDIÇÕES INDIVIDUAIS")
    coli = raw[raw["PARAMETRO NOME"] == TARGET_COL].copy()

    coli["DATE_KEY"] = pd.to_datetime(
        coli["DATA AMOSTRAGEM"]
    ).dt.strftime("%Y-%m-%d")

    coli["OPERADOR"] = coli["OPERADOR"].astype(str).str.strip()
    coli["VALUE"] = coli["RESULTADO"].apply(parse_result)

    # Yazarlarin yontemi: isareti silip sayiyi kullan.
    coli["VALUE_AUTHORS"] = coli["VALUE"]

    # Duzeltilmis yontem: '<x' = tespit siniri alti = bakteri bulunmadi.
    coli["VALUE_CORRECTED"] = np.where(
        coli["OPERADOR"] == "<", 0.0, coli["VALUE"]
    )

    # Sayim sonucu (N/100 mL) normalde tam sayidir; 0 ile 1 arasi
    # '=' sonuclari supheli olarak isaretlenir (danismana sorulacak).
    coli["IS_FRACTIONAL"] = (
        (coli["OPERADOR"] == "=")
        & (coli["VALUE"] > 0)
        & (coli["VALUE"] < 1)
    )

    coli["IS_INVESTIGATION"] = coli["CONTROL TYPE ANALISE"].astype(
        str
    ).str.contains(INVESTIGATION_TEXT)

    return coli


def join_unique(series):
    values = sorted({str(v).strip() for v in series if pd.notna(v)})
    return " | ".join(values)


def aggregate_per_sample(coli):
    grouped = coli.groupby(KEYS)

    agg = grouped.agg(
        META_N_COLIFORM_RECORDS=("VALUE", "size"),
        COLIFORM_AUTHORS=("VALUE_AUTHORS", "mean"),
        COLIFORM_CORRECTED=("VALUE_CORRECTED", "mean"),
        META_HAS_LESS_THAN=("OPERADOR", lambda s: int((s == "<").any())),
        META_HAS_GREATER_THAN=("OPERADOR", lambda s: int((s == ">").any())),
        META_FRACTIONAL_RESULT=("IS_FRACTIONAL", "max"),
        META_IS_INVESTIGATION=("IS_INVESTIGATION", "max"),
        META_CONTROL_TYPE=("CONTROL TYPE ANALISE", join_unique),
        META_LAB=("LAB NOME", join_unique),
        META_SOURCE_FILE=("FICHEIRO FONTE", join_unique),
    ).reset_index()

    agg["META_FRACTIONAL_RESULT"] = agg["META_FRACTIONAL_RESULT"].astype(int)
    agg["META_IS_INVESTIGATION"] = agg["META_IS_INVESTIGATION"].astype(int)

    agg["META_SAMPLING_TYPE"] = np.where(
        agg["META_IS_INVESTIGATION"] == 1,
        "Sorusturma",
        "Rutin",
    )

    return agg


def build():
    print("Ozellik tablosu :", FEATURES_CSV)
    print("Ham kayitlar    :", RAW_XLSX)

    features = pd.read_csv(FEATURES_CSV).copy()
    features["DATE_KEY"] = pd.to_datetime(
        features["DATA AMOSTRAGEM"]
    ).dt.strftime("%Y-%m-%d")

    coli = load_raw_coliform_records()
    samples = aggregate_per_sample(coli)

    merged = features.merge(
        samples, on=KEYS, how="left", validate="1:1"
    ).copy()

    labeled = merged[TARGET_COL].notna()

    # ---- Kontrol 1: yazarlarin etiketini ham veriden birebir uretebiliyor muyuz?
    authors_match = np.isclose(
        merged.loc[labeled, TARGET_COL],
        merged.loc[labeled, "COLIFORM_AUTHORS"],
        equal_nan=False,
    )
    if not authors_match.all():
        bad = merged.loc[labeled].loc[~authors_match, KEYS + [TARGET_COL]]
        raise ValueError(
            "Ozellik tablosundaki koliform degeri ham veriyle uyusmuyor:\n"
            f"{bad}"
        )

    unmatched = merged[merged["COLIFORM_AUTHORS"].notna() & ~labeled]
    if len(unmatched):
        raise ValueError(
            f"Ham veride olup ozellik tablosunda etiketsiz kalan "
            f"{len(unmatched)} numune var."
        )

    # ---- Duzeltme: hedef sutunu duzeltilmis degerle degistir.
    merged["META_COLIFORM_ORIGINAL"] = merged[TARGET_COL]
    merged.loc[labeled, TARGET_COL] = merged.loc[labeled, "COLIFORM_CORRECTED"]

    old_label = (merged["META_COLIFORM_ORIGINAL"] > 0).astype(float)
    new_label = (merged[TARGET_COL] > 0).astype(float)
    merged["META_LABEL_CHANGED"] = np.where(
        labeled, (old_label != new_label).astype(int), np.nan
    )

    # ---- Analiz icin yardimci bilgiler (modele girmez)
    dates = pd.to_datetime(merged["DATA AMOSTRAGEM"])
    merged["META_YEAR"] = dates.dt.year
    merged["META_MUNICIPALITY"] = merged["FREGUESIA"]
    merged["META_CHLORINE_MISSING"] = (
        merged["Desinfetante residual"].isna().astype(int)
    )

    merged = merged.drop(
        columns=["DATE_KEY", "COLIFORM_AUTHORS", "COLIFORM_CORRECTED"]
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(PROCESSED_CSV, index=False)

    changes = merged.loc[
        merged["META_LABEL_CHANGED"] == 1,
        [
            "FREGUESIA", "ZAPE NOME", "DATA AMOSTRAGEM",
            "META_COLIFORM_ORIGINAL", TARGET_COL,
            "META_N_COLIFORM_RECORDS", "META_CONTROL_TYPE",
            "META_SOURCE_FILE",
        ],
    ]
    changes.to_csv(DATA_AUDIT_DIR / "label_changes.csv", index=False)

    # ---- Ozet
    y_old = old_label[labeled]
    y_new = new_label[labeled]
    print("\n=== ISLENMIS VERI OZETI ===")
    print("Toplam satir (tum numune olaylari):", len(merged))
    print("Etiketli satir                    :", int(labeled.sum()))
    print("Ham koliform kaydi                :", len(coli))
    print(
        "Ayni ZAPE+gun icin birlestirilen fazla kayit:",
        int((merged.loc[labeled, "META_N_COLIFORM_RECORDS"] - 1).sum()),
    )
    print(f"Pozitif (eski etiket)             : {int(y_old.sum())}")
    print(f"Pozitif (duzeltilmis etiket)      : {int(y_new.sum())}")
    print("Degisen etiket sayisi             :", len(changes))
    if len(changes):
        print(changes.to_string(index=False))
    print(
        "Supheli kesirli sonuc (0<x<1, '=') :",
        int(merged.loc[labeled, "META_FRACTIONAL_RESULT"].sum()),
    )
    print(
        "Sorusturma numunesi               :",
        int(merged.loc[labeled, "META_IS_INVESTIGATION"].sum()),
    )
    print("\nKaydedildi:", PROCESSED_CSV)
    print("Kaydedildi:", DATA_AUDIT_DIR / "label_changes.csv")

    return merged


if __name__ == "__main__":
    build()
