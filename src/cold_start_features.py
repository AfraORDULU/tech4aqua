
# Cold-start icin iki farkli ozellik senaryosu

# Gecmis bolge olcumlerine veya hesaplama
# yontemi henuz dogrulanmamis ozelliklere
# dayanan degiskenler
RESTRICTED_FEATURES = {
    "DECAIMENTO_MICROBIO_PREV",
    "CLORO_RESIDUAL_PREV",
    "DELTA_DIAS_ZAPE",
    "LOG_DELTA_DIAS",
    "INDICE_ESTRESSE_CONJUNTO",
    "INDICE_RUNOFF_TURVACAO",
}

def build_cold_start_sets(X):
    """
    Mevcut model ozelliklerinden iki senaryo olusturur.

    basic: Daha ihtiyatli ozellik kumesi.
    extended: Mevcut temizleme hattinin tum ozellikleri.

    Not: Bu fonksiyon veri sizintisini tek basina
    dogrulamaz.
    """

    basic_columns = [
        col for col in X.columns
        if col not in RESTRICTED_FEATURES
    ]

    basic = X[basic_columns].copy()
    extended = X.copy()

    return basic, extended

def build_conservative_set(X):
    """
    Daha az turetilmis ozellik kullanan
    ihtiyatli deney senaryosu.
    """

    conservative_columns = [
        "Desinfetante residual",
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "MES",
    ]

    missing = [
        col for col in conservative_columns
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Eksik ozellikler: {missing}"
        )

    return X[conservative_columns].copy()
