"""Ablasyon için açık özellik grupları.

Sütun adında alt dizi araması yapılmaz. Listeler yalnızca mevcut
valid_numeric kümesiyle kesiştirilir; eksik sütun sessizce gruba eklenmez.
"""

# Beklenen 29 özellik: yalnızca bu isimler, alt dizi kuralı yok.
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
        "DECAIMENTO_MICROBIO_PREV",
        "FLAG_CALOR_EXTREMO",
        "FLAG_RUNOFF_RISCO",
        "INDICE_ESTRESSE_CONJUNTO",
        "INDICE_RUNOFF_TURVACAO",
    ],
}

GROUP_ORDER = ("Fizikokimya", "Meteoroloji", "Takvim")


def build_feature_groups(valid_numeric):
    """valid_numeric içindeki sütunları açık listelerle grupla ve doğrula."""
    valid_list = list(valid_numeric)
    valid_set = set(valid_list)

    feature_groups = {}
    missing_expected = {}
    for group_name in GROUP_ORDER:
        canonical = CANONICAL_FEATURE_GROUPS[group_name]
        present = [col for col in canonical if col in valid_set]
        absent = [col for col in canonical if col not in valid_set]
        feature_groups[group_name] = present
        if absent:
            missing_expected[group_name] = absent

    membership = {}
    for group_name, cols in feature_groups.items():
        for col in cols:
            membership.setdefault(col, []).append(group_name)

    duplicates = {col: groups for col, groups in membership.items() if len(groups) > 1}
    unassigned = [col for col in valid_list if col not in membership]
    unexpected_in_groups = [col for col in membership if col not in valid_set]
    union = [col for col in valid_list if col in membership]

    errors = []
    if missing_expected:
        errors.append(
            "Beklenen sütunlar valid_numeric içinde yok (gruba eklenmedi): "
            f"{missing_expected}"
        )
    if unexpected_in_groups:
        errors.append(
            "Grup listesinde olup valid_numeric içinde olmayan sütunlar: "
            f"{unexpected_in_groups}"
        )
    if duplicates:
        errors.append(f"Birden fazla gruba atanan sütunlar: {duplicates}")
    if unassigned:
        errors.append(f"Hiçbir gruba atanmamış sütunlar: {unassigned}")
    if set(union) != valid_set:
        errors.append(
            "Üç grubun birleşimi valid_numeric ile örtüşmüyor. "
            f"Birleşim={len(union)}, valid_numeric={len(valid_list)}"
        )

    if errors:
        raise ValueError(
            "Özellik grupları geçersiz:\n- " + "\n- ".join(errors)
        )

    return feature_groups


def make_ablation_scenarios(valid_numeric, feature_groups):
    """Tam model ve her grubun çıkarıldığı özellik kümeleri."""
    valid_list = list(valid_numeric)
    scenarios = {
        f"1. Tam Model (Tüm Özellikler - {len(valid_list)})": list(valid_list),
    }
    for index, group_name in enumerate(GROUP_ORDER, start=2):
        dropped = feature_groups[group_name]
        remaining = [col for col in valid_list if col not in dropped]
        overlap = [col for col in dropped if col not in valid_list]
        if overlap:
            raise ValueError(
                f"{group_name} grubunda valid_numeric dışında sütun var: {overlap}"
            )
        if len(remaining) != len(valid_list) - len(dropped):
            raise ValueError(
                f"{group_name} çıkarılınca özellik sayısı tutarsız: "
                f"kalan={len(remaining)}, beklenen={len(valid_list) - len(dropped)}"
            )
        scenarios[
            f"{index}. {group_name} Çıkarıldı (-{len(dropped)})"
        ] = remaining
    return scenarios
