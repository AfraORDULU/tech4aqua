
"""Ablasyon icin acik ozellik gruplari.

Sutun adinda alt dizi aramasi yapilmaz.
Listeler yalnizca mevcut valid_numeric kumesiyle kesisitirilir.
Eksik sutunlar sessizce gruba eklenmez.
"""

# Beklenen 28 ozellik: yalnizca bu isimler kullanilir.
# DECAIMENTO_MICROBIO_PREV veri sizintisi riski nedeniyle
# ozellik listesinden cikarilmistir.

CANONICAL_FEATURE_GROUPS = {
    "Takvim": [
        "MES",
        "SIN_MES",
        "COS_MES",
        "FLAG_VERAO",
    ],

    "Meteoroloji": [
        "TEMP MÁX (ºC)",
        "PRECIPITAÇÃO (mm)",
        "UMIDADE MÉDIA (%)",
        "PRECIP_LAG1_REGIONAL",
        "PRECIP_DIFF_REGIONAL",
        "TEMP_LAG1_REGIONAL",
        "TEMP_DIFF_REGIONAL",
        "PRECIP_EWMA_REGIONAL",
        "TEMP_EWMA_REGIONAL",
    ],

    "Fizikokimya": [
        "Desinfetante residual",
        "FLAG_PH_ANORMAL",
        "FLAG_CLORO_CRITICO",
        "FLAG_TURVACAO_ALTA",
        "CLORO_RESIDUAL_PREV",
        "DEGRADACAO_TERMICA_CLORO",
        "ZAPE VOLUME",
        "ZAPE POPULACAO",
        "DELTA_DIAS_ZAPE",
        "LOG_DELTA_DIAS",
        "Proxy_Tempo_Retencao",
        "FLAG_CALOR_EXTREMO",
        "FLAG_RUNOFF_RISCO",
        "INDICE_ESTRESSE_CONJUNTO",
        "INDICE_RUNOFF_TURVACAO",
    ],
}

GROUP_ORDER = (
    "Fizikokimya",
    "Meteoroloji",
    "Takvim",
)


def build_feature_groups(valid_numeric):
    """valid_numeric icindeki sutunlari grupla ve dogrula."""

    valid_list = list(valid_numeric)
    valid_set = set(valid_list)

    feature_groups = {}
    missing_expected = {}

    for group_name in GROUP_ORDER:
        canonical = CANONICAL_FEATURE_GROUPS[group_name]

        present = [
            col for col in canonical
            if col in valid_set
        ]

        absent = [
            col for col in canonical
            if col not in valid_set
        ]

        feature_groups[group_name] = present

        if absent:
            missing_expected[group_name] = absent

    membership = {}

    for group_name, cols in feature_groups.items():
        for col in cols:
            membership.setdefault(col, []).append(group_name)

    duplicates = {
        col: groups
        for col, groups in membership.items()
        if len(groups) > 1
    }

    unassigned = [
        col for col in valid_list
        if col not in membership
    ]

    unexpected_in_groups = [
        col for col in membership
        if col not in valid_set
    ]

    union = [
        col for col in valid_list
        if col in membership
    ]

    errors = []

    if missing_expected:
        errors.append(
            "Beklenen sutunlar valid_numeric icinde yok "
            "(gruba eklenmedi): "
            f"{missing_expected}"
        )

    if unexpected_in_groups:
        errors.append(
            "Grup listesinde olup valid_numeric icinde "
            "olmayan sutunlar: "
            f"{unexpected_in_groups}"
        )

    if duplicates:
        errors.append(
            f"Birden fazla gruba atanan sutunlar: {duplicates}"
        )

    if unassigned:
        errors.append(
            f"Hicbir gruba atanmamis sutunlar: {unassigned}"
        )

    if set(union) != valid_set:
        errors.append(
            "Uc grubun birlesimi valid_numeric ile ortusmuyor. "
            f"Birlesim={len(union)}, "
            f"valid_numeric={len(valid_list)}"
        )

    if errors:
        raise ValueError(
            "Ozellik gruplari gecersiz:\n- "
            + "\n- ".join(errors)
        )

    return feature_groups


def make_ablation_scenarios(valid_numeric, feature_groups):
    """Tam model ve her grubun cikarildigi ozellik kumeleri."""

    valid_list = list(valid_numeric)

    scenarios = {
        f"1. Tam Model (Tum Ozellikler - {len(valid_list)})":
            list(valid_list),
    }

    for index, group_name in enumerate(GROUP_ORDER, start=2):

        dropped = feature_groups[group_name]

        remaining = [
            col for col in valid_list
            if col not in dropped
        ]

        overlap = [
            col for col in dropped
            if col not in valid_list
        ]

        if overlap:
            raise ValueError(
                f"{group_name} grubunda valid_numeric "
                f"disinda sutun var: {overlap}"
            )

        if len(remaining) != len(valid_list) - len(dropped):
            raise ValueError(
                f"{group_name} cikarilinca ozellik "
                f"sayisi tutarsiz: kalan={len(remaining)}, "
                f"beklenen={len(valid_list) - len(dropped)}"
            )

        scenarios[
            f"{index}. {group_name} Cikarildi (-{len(dropped)})"
        ] = remaining

    return scenarios
