"""Cold-start icin ozellik grubu senaryolari."""

FEATURE_GROUPS = {
    "Guncel_Olcumler": [
        "Desinfetante residual",
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
    ],
    "Takvim": [
        "MES",
        "SIN_MES",
        "COS_MES",
        "FLAG_VERAO",
    ],
    "Bolgesel": [
        "ZAPE VOLUME",
        "ZAPE POPULACAO",
        "Proxy_Tempo_Retencao",
    ],
    "Gecmis_ve_Degisim": [
        "CLORO_RESIDUAL_PREV",
        "DELTA_DIAS_ZAPE",
        "LOG_DELTA_DIAS",
        "PRECIP_LAG1_REGIONAL",
        "PRECIP_DIFF_REGIONAL",
        "TEMP_LAG1_REGIONAL",
        "TEMP_DIFF_REGIONAL",
        "PRECIP_EWMA_REGIONAL",
        "TEMP_EWMA_REGIONAL",
    ],
    "Risk_Bayraklari": [
        "FLAG_CLORO_CRITICO",
        "FLAG_TURVACAO_ALTA",
        "FLAG_PH_ANORMAL",
        "FLAG_CALOR_EXTREMO",
        "FLAG_RUNOFF_RISCO",
    ],
    "Bilesik_Gostergeler": [
        "INDICE_ESTRESSE_CONJUNTO",
        "INDICE_RUNOFF_TURVACAO",
        "DEGRADACAO_TERMICA_CLORO",
    ],
}


def build_ablation_sets(X):
    """Mevcut sutunlardan deney senaryolari olustur."""

    all_group_columns = [
        col
        for columns in FEATURE_GROUPS.values()
        for col in columns
    ]

    if len(all_group_columns) != len(set(all_group_columns)):
        raise ValueError("Bir ozellik birden fazla grupta!")

    missing = [
        col for col in all_group_columns
        if col not in X.columns
    ]

    unexpected = [
        col for col in X.columns
        if col not in all_group_columns
    ]

    if missing or unexpected:
        raise ValueError(
            f"Eksik ozellikler: {missing}\n"
            f"Gruplanmamis ozellikler: {unexpected}"
        )

    conservative = [
        "Desinfetante residual",
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "MES",
    ]

    previous_weather = [
        "PRECIP_LAG1_REGIONAL",
        "PRECIP_DIFF_REGIONAL",
        "TEMP_LAG1_REGIONAL",
        "TEMP_DIFF_REGIONAL",
        "PRECIP_EWMA_REGIONAL",
        "TEMP_EWMA_REGIONAL",
    ]

    regional_features = [
        "ZAPE VOLUME",
        "ZAPE POPULACAO",
        "Proxy_Tempo_Retencao",
    ]

    previous_zape_features = [
        "CLORO_RESIDUAL_PREV",
        "DELTA_DIAS_ZAPE",
        "LOG_DELTA_DIAS",
    ]

    scenarios = {
        "A_Ihtiyatli": conservative,

        "B_Klorsuz": [
            col for col in conservative
            if col != "Desinfetante residual"
        ],

        "C_Bolgesel": (
            conservative + regional_features
        ),

        "D_Gecmis_Meteoroloji": (
            conservative + previous_weather
        ),

        "E_Risk_Bayraklari": (
            conservative + FEATURE_GROUPS["Risk_Bayraklari"]
        ),

        "F_Genisletilmis": list(X.columns),

        # YENI SENARYO G
        # Bolgesel ozellikleri cikar.
        "G_Bolgesiz": [
            col for col in X.columns
            if col not in regional_features
        ],

        # YENI SENARYO H
        # Bolgesel ve gecmis ZAPE ozelliklerini cikar.
        "H_Bolgesiz_Gecmissiz": [
            col for col in X.columns
            if col not in (
                regional_features + previous_zape_features
            )
        ],
    }

    result = {}

    for name, columns in scenarios.items():

        if len(columns) != len(set(columns)):
            raise ValueError(
                f"{name} senaryosunda tekrar eden sutun var!"
            )

        result[name] = X[columns].copy()

    return result