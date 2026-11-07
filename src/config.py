"""
Tech4Aqua - Ortak dosya yollari.

Butun betikler veri yolunu buradan alir. Boylece veri klasoru
farkli bilgisayarlarda farkli yerde olsa bile tek yerden ayarlanir.

Kullanim (proje kok klasorunden):
    python -m src.build_dataset          # islenmis veriyi bir kez uret
    python final_rf_experiments.py       # diger betikler DATA_PATH'i kullanir
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_AUDIT_DIR = RESULTS_DIR / "data_audit"


def _first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# Mendeley paketinin orijinal klasor yapisi: data/raw/01_Data/...
# Eski kurulumlarda dosyalar dogrudan data/raw/ altinda olabilir.
FEATURES_CSV = _first_existing([
    DATA_DIR / "raw" / "01_Data" / "Base_Tech4Aqua_Features_Avancadas.csv",
    DATA_DIR / "raw" / "Base_Tech4Aqua_Features_Avancadas.csv",
])

RAW_XLSX = _first_existing([
    DATA_DIR / "raw" / "01_Data" / "Base_Revisada_Tech4Aqua.xlsx",
    DATA_DIR / "raw" / "Base_Revisada_Tech4Aqua.xlsx",
])

DICTIONARY_CSV = _first_existing([
    DATA_DIR / "raw" / "01_Data" / "Data_Dictionary.csv",
    DATA_DIR / "raw" / "Data_Dictionary.csv",
])

# src/build_dataset.py tarafindan uretilen, etiketi duzeltilmis
# ve ornekleme bilgisi (META_ sutunlari) eklenmis tablo.
PROCESSED_CSV = PROCESSED_DIR / "tech4aqua_labeled.csv"

# Modelleme betiklerinin kullandigi veri:
# islenmis tablo varsa o, yoksa orijinal ozellik tablosu.
DATA_PATH = PROCESSED_CSV if PROCESSED_CSV.exists() else FEATURES_CSV
