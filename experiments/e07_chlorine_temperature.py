"""E07: LR ile klor, sicaklik ve turetilmis ozellik karsilastirmasi.

Calistirma (proje kokunden):
    python -m experiments.e07_chlorine_temperature

Tum modeller: ayni 5 tohum, 5 katli ZAPE gruplu, arindirilmis CV.
Egitim tum numunelerle; raporlama Tumu ve Rutin alt kumelerinde.
Eksik deger islemleri mevcut model altyapisina birakilir.
"""

import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data import CHLORINE, FEATURE_SETS, load_dataset
from src.evaluation import mean_std, run_cv, summarize


OUT = RESULTS_DIR / "e07_chlorine_temperature"
SEEDS = (0, 1, 2, 3, 4)
TEMP = "TEMP MÁX (ºC)"
THERMAL = "DEGRADACAO_TERMICA_CLORO"


def main():
    df = load_dataset().reset_index(drop=True)
    feature_sets = {
        "M1_klor": [CHLORINE],
        "M2_klor_sicaklik": [CHLORINE, TEMP],
        "M3_klor_sicaklik_termal": [CHLORINE, TEMP, THERMAL],
        "M4_C": list(FEATURE_SETS["C"]),
    }
    features = list(dict.fromkeys(
        col for cols in feature_sets.values() for col in cols
    ))
    missing = sorted(set(features + ["y", "is_investigation"]) - set(df))
    if missing:
        raise ValueError(f"Eksik sutunlar: {missing}")
    for name, cols in feature_sets.items():
        if not cols or len(cols) != len(set(cols)):
            raise ValueError(f"Bos veya tekrarlanan ozellik listesi: {name}")
    if df.empty or df["y"].isna().any() or set(df["y"].unique()) != {0, 1}:
        raise ValueError("y eksiksiz olmali ve hem 0 hem 1 icermeli.")
    flag = df["is_investigation"]
    if flag.isna().any() or not flag.isin([False, True]).all():
        raise ValueError("is_investigation eksiksiz bir boolean/0-1 olmali.")
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Sayisal olmayan ozellik: {col}")
        values = df[col].dropna().to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Tamamen eksik veya sonsuz degerli ozellik: {col}")

    # Mevcut turetilmis sutunu dogrula; yeniden hesaplayip degistirme.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = df[TEMP] / (df[CHLORINE] + 0.1)
    valid = expected.notna() & df[THERMAL].notna()
    if not valid.any() or not np.allclose(
        df.loc[valid, THERMAL], expected.loc[valid], rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Termal ozellik TEMP / (CHLORINE + 0.1) ile uyusmuyor.")

    subsets = {
        "Tumu": np.ones(len(df), dtype=bool),
        "Rutin": ~flag.to_numpy(dtype=bool),
    }
    for name, mask in subsets.items():
        if df.loc[mask, "y"].nunique() != 2:
            raise ValueError(f"{name} alt kumesi her iki sinifi da icermeli.")

    OUT.mkdir(parents=True, exist_ok=True)
    df[features].isna().sum().rename("n_missing").to_csv(
        OUT / "missing_values.csv", index_label="feature"
    )
    metadata = {
        "model": "LR", "seeds": list(SEEDS), "n_splits": 5,
        "purge": True, "tune_threshold": True,
        "training": "all_samples", "feature_sets": feature_sets,
        "n_rows": len(df), "formula_checked_rows": int(valid.sum()),
    }
    (OUT / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_rows, seed_tables = [], []
    reference_folds = None
    for name, cols in feature_sets.items():
        print(f"Calisiyor: {name} ({len(cols)} ozellik)", flush=True)
        oof = run_cv(
            df, cols, "LR", seeds=SEEDS, n_splits=5,
            purge=True, train_mask=None, tune_threshold=True,
        ).sort_values(["seed", "row"]).reset_index(drop=True)
        expected_pairs = pd.MultiIndex.from_product(
            [SEEDS, range(len(df))], names=["seed", "row"]
        )
        actual_pairs = pd.MultiIndex.from_frame(oof[["seed", "row"]])
        if not actual_pairs.equals(expected_pairs):
            raise ValueError(f"{name}: her satir/tohum icin tek OOF tahmini gerekli.")
        if not np.isfinite(oof[["prob", "threshold"]].to_numpy()).all():
            raise ValueError(f"{name}: gecersiz olasilik veya esik.")
        if not oof["prob"].between(0, 1).all():
            raise ValueError(f"{name}: olasilik [0, 1] disinda.")
        folds = oof[["seed", "row", "fold"]]
        if reference_folds is None:
            reference_folds = folds.copy()
        elif not folds.equals(reference_folds):
            raise ValueError("Ozellik gruplarinin CV bolmeleri farkli.")
        oof.to_csv(OUT / f"oof_{name}.csv", index=False)
        for subset_name, mask in subsets.items():
            per_seed = summarize(df, oof, subset=mask)
            tags = {"feature_set": name, "model": "LR", "test": subset_name}
            seed_tables.append(per_seed.assign(**tags))
            summary_rows.append({**tags, **mean_std(per_seed)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(seed_tables, ignore_index=True).to_csv(
        OUT / "per_seed.csv", index=False
    )
    print(summary.round(4).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
