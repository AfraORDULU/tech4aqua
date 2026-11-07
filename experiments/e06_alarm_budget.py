"""E06 - Rutin numunelerde alarm butcesi ve klor esitlik analizi.

Calistirma:
    python -m experiments.e06_alarm_budget

LR:
    Purged / A / LR OOF tahminleri, mevcut bes seed uzerinden
    degerlendirilir. Esit LR skorlari orijinal satir sirasiyla cozulur.

Yalniz klor:
    Dusuk klor daha yuksek risk kabul edilir.
    Esit klor degerleri tekrarlanabilir rastgele siralamayla cozulur.
    Her tekrarda ayni siralamanin ilk k elemani kullanildigindan,
    %5, %10 ve %20 alarm kumeleri birbirinin alt kumesidir.

Yorum:
    Klor tekrarlarinin tamami AYNI rutin numunelere aittir.
    Tekrarlar bagimsiz veri setleri veya model seed'leri degildir.
    Klor yuzdelikleri, yalnizca esitlik cozumunden kaynaklanan
    degiskenligi gosterir; populasyon guven araligi degildir.

Eski alarm_budget_results.csv ve alarm_budget_summary.csv korunur.
Yeni sonuclar v2_ on ekiyle yazilir.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.config import RESULTS_DIR
from src.data import CHLORINE, load_dataset


OUT = RESULTS_DIR / "e06_alarm_budget"
OOF_PATH = RESULTS_DIR / "e01_baseline" / "oof.csv"

BUDGETS = (0.05, 0.10, 0.20)
EXPECTED_LR_SEEDS = 5

TIE_RANDOM_SEED = 20261107
N_TIE_REPEATS = 10_000

METRICS = (
    "dogru_alarm",
    "yanlis_alarm",
    "precision",
    "recall",
    "MCC",
)


def alarm_count(n, budget):
    """Butceyi asagi yuvarlamadan alarm sayisina cevir."""
    if n < 1 or not 0 < budget <= 1:
        raise ValueError("Numune sayisi ve alarm butcesi gecersiz.")
    return max(1, int(np.ceil(n * budget)))


def evaluate_order(y, order, budget):
    """Verilen risk siralamasinin ilk k numunesini sec."""
    k = alarm_count(len(y), budget)

    pred = np.zeros(len(y), dtype=int)
    pred[order[:k]] = 1

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = k - tp
    positives = int(y.sum())

    return {
        "numune": len(y),
        "pozitif": positives,
        "alarm_sayisi": k,
        "dogru_alarm": tp,
        "yanlis_alarm": fp,
        "precision": tp / k,
        "recall": tp / positives if positives else np.nan,
        "MCC": matthews_corrcoef(y, pred),
    }


def boundary_info(scores, budget):
    """Alarm sinirindaki esit skor grubunu tanimla."""
    k = alarm_count(len(scores), budget)
    cutoff = np.sort(scores)[::-1][k - 1]

    above = scores > cutoff
    tied = scores == cutoff

    return {
        "sinir_skoru": float(cutoff),
        "sinirdan_onceki_numune": int(above.sum()),
        "sinirdaki_esit_numune": int(tied.sum()),
        "esit_gruptan_secilecek": k - int(above.sum()),
    }


def random_tie_order(scores, rng):
    """Skor onceligini koru; yalnizca esit skorlarin sirasini karistir.

    Once tum satirlari rastgele permute edip ardindan kararli
    siralama yapmak, her esit skor grubuna rastgele bir sira verir.
    Skorlara yapay gurultu eklenmez; farkli skorlar yer degistirmez.
    """
    shuffled = rng.permutation(len(scores))
    return shuffled[
        np.argsort(-scores[shuffled], kind="stable")
    ]


def load_inputs():
    """Veriyi ve bes seed'e ait tam OOF tahminlerini dogrula."""
    df = load_dataset()

    required_data = {"y", "is_investigation", CHLORINE}
    missing = required_data.difference(df.columns)
    if missing:
        raise ValueError(f"Veride eksik sutunlar: {sorted(missing)}")

    investigation = df["is_investigation"]
    if (
        investigation.isna().any()
        or not investigation.isin([True, False]).all()
    ):
        raise ValueError(
            "is_investigation eksiksiz boolean veya 0/1 olmali."
        )

    routine = ~investigation.to_numpy(dtype=bool)
    if not routine.any():
        raise ValueError("Rutin numune bulunamadi.")

    labels = pd.to_numeric(df["y"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("y sutunu eksiksiz ve yalnizca 0/1 olmali.")

    y = labels.to_numpy(dtype=int)[routine]

    # Yalnizca analiz edilen rutin numunelerin klorunu kullan.
    chlorine = pd.to_numeric(
        df.loc[routine, CHLORINE], errors="raise"
    ).to_numpy(dtype=float)

    # Mevcut veride rutin klor eksigi yok. Veri degisirse sessizce
    # medyan atamak yerine acik bir hata ver.
    if not np.isfinite(chlorine).all():
        raise ValueError(
            "Rutin numunelerde eksik veya sonsuz klor degeri var. "
            "Analizden once eksik veri politikasini belirleyin."
        )

    oof = pd.read_csv(OOF_PATH)
    required_oof = {
        "protocol", "feature_set", "model", "seed", "row", "prob"
    }
    missing = required_oof.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF'ta eksik sutunlar: {sorted(missing)}")

    oof = oof.loc[
        (oof["protocol"] == "purged")
        & (oof["feature_set"] == "A")
        & (oof["model"] == "LR")
    ].copy()

    if oof.empty or oof["seed"].isna().any():
        raise ValueError("Gecerli purged / A / LR seed'leri bulunamadi.")

    if oof["seed"].nunique() != EXPECTED_LR_SEEDS:
        raise ValueError(
            f"{EXPECTED_LR_SEEDS} LR seed'i bekleniyordu; "
            f"{oof['seed'].nunique()} bulundu."
        )

    oof["row"] = pd.to_numeric(oof["row"], errors="raise")
    oof["prob"] = pd.to_numeric(oof["prob"], errors="raise")

    model_scores = {}
    for seed, predictions in oof.groupby("seed", sort=True):
        predictions = predictions.sort_values("row")

        if (
            len(predictions) != len(df)
            or predictions["row"].duplicated().any()
            or not np.array_equal(
                predictions["row"].to_numpy(),
                np.arange(len(df)),
            )
        ):
            raise ValueError(
                f"Seed {seed}: eksik, tekrarli veya gecersiz row var."
            )

        probabilities = predictions["prob"].to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError(
                f"Seed {seed}: prob degerleri sonlu ve [0, 1] olmali."
            )

        model_scores[seed] = probabilities[routine]

    return y, chlorine, model_scores


def summarize_lr(results):
    """Ayni veri uzerindeki bes model seed'inin degiskenligi."""
    rows = []

    for budget, group in results.groupby("alarm_butcesi", sort=True):
        row = {
            "yontem": "LR_A",
            "alarm_butcesi": budget,
            "seed_sayisi": len(group),
            "numune": int(group["numune"].iloc[0]),
            "pozitif": int(group["pozitif"].iloc[0]),
            "alarm_sayisi": int(group["alarm_sayisi"].iloc[0]),
        }

        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std_seed"] = group[metric].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_chlorine(results, y, scores):
    """Sabit veri uzerindeki klor esitlik belirsizligini ozetle."""
    rows = []

    for budget, group in results.groupby("alarm_butcesi", sort=True):
        boundary = boundary_info(scores, budget)
        cutoff = boundary["sinir_skoru"]

        above = scores > cutoff
        tied = scores == cutoff
        slots = boundary["esit_gruptan_secilecek"]
        tied_n = int(tied.sum())
        tied_positive = int(y[tied].sum())
        fixed_tp = int(y[above].sum())

        # Esit gruptan yerine koymadan, uniform secimin tam beklentisi.
        expected_tp = fixed_tp + slots * tied_positive / tied_n

        # Tum olasi esitlik cozumleri altinda ulasilabilir TP sinirlari.
        min_tp = fixed_tp + max(0, slots - (tied_n - tied_positive))
        max_tp = fixed_tp + min(slots, tied_positive)

        row = {
            "yontem": "Yalniz_klor",
            "alarm_butcesi": budget,
            "numune": len(y),
            "pozitif": int(y.sum()),
            "alarm_sayisi": alarm_count(len(y), budget),
            "tie_tekrar_sayisi": len(group),
            "tie_random_seed": TIE_RANDOM_SEED,
            "sinir_klor_mg_L": -cutoff,
            "sinirdan_onceki_numune":
                boundary["sinirdan_onceki_numune"],
            "sinirdaki_esit_numune": tied_n,
            "esit_gruptaki_pozitif": tied_positive,
            "esit_gruptan_secilecek": slots,
            "dogru_alarm_exact_expected": expected_tp,
            "dogru_alarm_possible_min": min_tp,
            "dogru_alarm_possible_max": max_tp,
        }

        for metric in METRICS:
            values = group[metric]
            row[f"{metric}_mc_mean"] = values.mean()
            row[f"{metric}_tie_std"] = values.std(ddof=1)
            row[f"{metric}_tie_p025"] = values.quantile(0.025)
            row[f"{metric}_tie_p975"] = values.quantile(0.975)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    y, chlorine, model_scores = load_inputs()
    chlorine_scores = -chlorine

    # LR: her seed bir kez, her butcede ayni skor siralamasi.
    lr_rows = []
    for seed, scores in model_scores.items():
        order = np.argsort(-scores, kind="stable")

        for budget in BUDGETS:
            boundary = boundary_info(scores, budget)
            lr_rows.append({
                "seed": seed,
                "yontem": "LR_A",
                "alarm_butcesi": budget,
                "tie_policy": "stable_original_row",
                **boundary,
                **evaluate_order(y, order, budget),
            })

    lr_results = pd.DataFrame(lr_rows)
    lr_summary = summarize_lr(lr_results)

    # Klor, LR seed dongusunun DISINDA hesaplanir.
    # Ayni klor sonuclari bes kez cogaltilmaz.
    rng = np.random.default_rng(TIE_RANDOM_SEED)
    chlorine_rows = []

    for repeat in range(N_TIE_REPEATS):
        order = random_tie_order(chlorine_scores, rng)

        for budget in BUDGETS:
            chlorine_rows.append({
                "tie_repeat": repeat,
                "tie_random_seed": TIE_RANDOM_SEED,
                "yontem": "Yalniz_klor",
                "alarm_butcesi": budget,
                **evaluate_order(y, order, budget),
            })

    chlorine_results = pd.DataFrame(chlorine_rows)
    chlorine_summary = summarize_chlorine(
        chlorine_results, y, chlorine_scores
    )

    # Eski CSV adlarina dokunulmaz.
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "v2_lr_seed_results.csv": lr_results,
        "v2_lr_seed_summary.csv": lr_summary,
        "v2_chlorine_tie_results.csv": chlorine_results,
        "v2_chlorine_tie_summary.csv": chlorine_summary,
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)

    print("\n=== E06: RUTIN NUMUNELERDE ALARM BUTCESI ===")
    print(f"Rutin numune: {len(y)} | Pozitif: {int(y.sum())}")
    print(f"LR seed sayisi: {len(model_scores)}")
    print(
        f"Klor esitlik tekrari: {N_TIE_REPEATS} | "
        f"Rastgelelik seed'i: {TIE_RANDOM_SEED}"
    )

    print("\n--- LR: bes seed ortalamasi ve seed standart sapmasi ---")
    print(lr_summary.round(4).to_string(index=False))

    print("\n--- KLOR: sabit veri uzerinde esitlik belirsizligi ---")
    print(chlorine_summary.round(4).to_string(index=False))

    print(
        "\nNOT: Klor tie_p025 / tie_p975 degerleri, rastgele esitlik "
        "cozumlerinin %2.5 ve %97.5 yuzdelikleridir. "
        "Populasyon guven araligi degildir."
    )
    print(
        "LR seed'leri ve klor tekrarlarinin tamami ayni rutin "
        "numuneleri kullanir. Bagimsiz veri setleri degildir; "
        "bu ozetlerden tek basina istatistiksel ustunluk cikarilmaz."
    )
    print(
        "LR skor esitlikleri orijinal satir sirasiyla cozulur. "
        "Sinirdaki esitlik gruplari LR sonuc dosyasinda raporlanir."
    )

    print(f"\nSonuclar: {OUT}")
    for filename in outputs:
        print(f"  {filename}")


if __name__ == "__main__":
    main()