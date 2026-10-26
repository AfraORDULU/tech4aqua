# src/final_feature_sets.py

"""
Tech4Aqua - Nihai deneyler için aday özellik senaryoları.

Bu senaryolar özelliklerin tahmin anında erişilebilir
olduğunu tek başına garanti etmez.
"""

def build_final_feature_sets(X):

    scenarios = {

        "A_Conservative": [
            "Desinfetante residual",
            "TEMP MÁX (ºC)",
            "PRECIPITAÇÃO (mm)",
            "UMIDADE MÉDIA (%)",
            "MES",
        ],

        "B_Meteorological": [
            "TEMP MÁX (ºC)",
            "PRECIPITAÇÃO (mm)",
            "UMIDADE MÉDIA (%)",
            "MES",
            "SIN_MES",
            "COS_MES",
            "FLAG_VERAO",
        ],

        "C_Extended": [
            "Desinfetante residual",
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
    }

    result = {}

    for name, columns in scenarios.items():

        missing = [
            col for col in columns
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"{name} senaryosunda eksik ozellikler: {missing}"
            )

        result[name] = X[columns].copy()

    return result