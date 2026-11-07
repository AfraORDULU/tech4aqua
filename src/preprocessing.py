import pandas as pd
import numpy as np

from src.feature_groups import build_feature_groups

META_PREFIX = "META_"


def load_metadata(file_path):
    """
    load_and_preprocess_data ile ayni satirlar (ayni index) icin
    analiz bilgilerini dondurur: belediye, tarih, ZAPE ve META_
    sutunlari. Duyarlilik analizleri (or. sorusturma numuneleri
    haric) ve yeni gruplama protokolleri icin kullanilir.
    """
    df = pd.read_csv(file_path)
    df_clean = df[df['Bactérias coliformes'].notna()]

    cols = ['FREGUESIA', 'ZAPE NOME', 'DATA AMOSTRAGEM'] + [
        c for c in df_clean.columns if c.startswith(META_PREFIX)
    ]
    return df_clean[cols].copy()

def load_and_preprocess_data(file_path):
    """
    Ham veriyi okur, sızıntı kaynaklarını temizler ve 
    ablasyon / modelleme için temiz özellik matrisini hazırlar.
    """
    # 1. Ham veriyi oku
    df = pd.read_csv(file_path)

    if not any(c.startswith(META_PREFIX) for c in df.columns):
        print(
            "UYARI: Islenmemis (orijinal) tablo kullaniliyor; 2 hatali "
            "'<' etiketi duzeltilmemis. Once calistirin: "
            "python -m src.build_dataset"
        )

    # 2. Sadece hedef ölçümü yapılmış (NaN olmayan) gerçek gözlem satırlarını filtrele
    df_clean = df[df['Bactérias coliformes'].notna()].copy()
    
    # 3. Hedef değişkeni ikili (binary) formata çevir (0: Temiz, 1: Riskli/Mevzuata Aykırı)
    df_clean['TARGET_BINARY'] = (df_clean['Bactérias coliformes'] > 0).astype(int)
    
    # 4. Sızıntı yapabilecek veya kimlik belirten yasaklı sütunları tanımla
    target_col = 'Bactérias coliformes'
    group_col = 'ZAPE NOME'
    id_cols = ['FREGUESIA', group_col, 'DATA AMOSTRAGEM']
    
    drop_leakage = [target_col, group_col, 'DATA AMOSTRAGEM', 'FREGUESIA', 'TARGET_BINARY']
    
    # Adında 'coli', 'bact', 'target' veya 'falha' geçen gizli sızıntı sütunlarını otomatik yakala
    auto_leakage = [
        c for c in df_clean.columns 
        if any(k in c.lower() for k in ['coli', 'bact', 'target', 'falha'])
    ]
    all_forbidden = set(drop_leakage + auto_leakage)
        # Cold-start: gecmis mikrobiyolojik bilgileri modele dahil etme
    cold_start_forbidden = [
        "DECAIMENTO_MICROBIO_PREV",
    ]

    all_forbidden.update(cold_start_forbidden)

    # META_ sutunlari (ornekleme turu, laboratuvar, yil, klor-eksik
    # bayragi vb.) yalnizca analiz icindir; modele asla girmez.
    all_forbidden.update(
        c for c in df_clean.columns if c.startswith(META_PREFIX)
    )
    
    # 5. Sayısal sütunları seç ve %50'den fazla eksik olanları / sabit sütunları ele
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    missing_ratios = df_clean[numeric_cols].isna().sum() / len(df_clean)
    
    valid_numeric = [
        c for c in numeric_cols 
        if c not in all_forbidden 
        and missing_ratios[c] < 0.50 
        and df_clean[c].nunique() > 1
    ]
    
    # 6. Özellik grupları (açık liste; src/feature_groups.py)
    feature_groups = build_feature_groups(valid_numeric)
    
    X = df_clean[valid_numeric]
    y = df_clean['TARGET_BINARY']
    groups = df_clean[group_col]
    
    return X, y, groups, feature_groups