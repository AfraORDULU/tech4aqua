"""
Deney 17 - Karar analizi: alarm butcesi, esik egrileri ve basari egrisi.

Yeni model egitilmez; E13, E14 ve E16'nin test-disi (ZAPE gruplu CV)
olasiliklari kullanilir (her numune icin 3 tohumun ortalamasi).

1) Alarm butcesi: Su idaresi numunelerin yalnizca %5, %10 veya %20'sine
   alarm verebilseydi, en riskli gorulen numunelere alarm verildiginde kac
   kirlilik yakalanirdi? Klor esitlikleri 2000 rastgele siralamanin
   ortalamasiyla cozulur.
2) Esik egrileri: TabPFN + gecmis olasiliklari icin farkli esiklerde alarm
   sayisi, kesinlik (precision) ve duyarlilik (recall).
3) Basari egrisi: bolgede sonucu bilinen gecmis numune sayisina gore
   ROC-AUC (E11 Sekil 8'in guncel modellerle tekrari).

Calistirma: python -m experiments.e17_decision_analysis
Ciktilar: results/e17_decision_analysis/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.config import RESULTS_DIR
from src.data import CHLORINE, load_dataset
from src.history import add_history_features

OUT = RESULTS_DIR / "e17_decision_analysis"
BUDGETS = (0.05, 0.10, 0.20)
N_TIE = 2000
BINS = [-1, 0, 2, 5, 10, 10_000]
BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11+"]
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def load_probs():
    """Model adi -> numune basina ortalama test-disi olasilik."""
    probs = {}
    e13 = pd.read_csv(RESULTS_DIR / "e13_tabpfn_history" / "oof.csv")
    e13 = e13[e13["protocol"] == "zape_cv"]
    names = {("LR", "A"): "A + LR (ana model)",
             ("LR", "A+HIST"): "A + gecmis + LR",
             ("TABPFN", "A"): "A + TabPFN",
             ("TABPFN", "A+HIST"): "A + gecmis + TabPFN"}
    for (m, fs), name in names.items():
        g = e13[(e13["model"] == m) & (e13["feature_set"] == fs)]
        probs[name] = g.groupby("row")["prob"].mean().sort_index()
    p16 = RESULTS_DIR / "e16_routine_history" / "oof.csv"
    if p16.exists():
        e16 = pd.read_csv(p16)
        g = e16[(e16["model"] == "TABPFN") & (e16["feature_set"] == "A+HISTR")]
        if len(g):
            probs["A + rutin gecmis + TabPFN"] = (
                g.groupby("row")["prob"].mean().sort_index())
    p14 = RESULTS_DIR / "e14_sequence_selection" / "oof.csv"
    if p14.exists():
        e14 = pd.read_csv(p14)
        e14 = e14[e14["protocol"] == "zape_cv"]
        for m in ("GRU (secimli)", "GRU dizi + ozet (secimli)"):
            g = e14[e14["model"] == m]
            if len(g):
                probs[m] = g.groupby("row")["prob"].mean().sort_index()
    return probs


def budget_table(y, scores, mask, rng):
    rows = []
    idx = np.where(mask)[0]
    yy = y[idx]
    n = len(idx)
    for name, s in scores.items():
        ss = s[idx]
        for b in BUDGETS:
            k = max(1, int(np.ceil(n * b)))
            if name == "Egitimsiz klor":
                tps = []
                for _ in range(N_TIE):
                    order = np.lexsort((rng.random(n), -ss))
                    tps.append(yy[order[:k]].sum())
                tp = float(np.mean(tps))
                lo, hi = np.percentile(tps, [2.5, 97.5])
            else:
                order = np.argsort(-ss, kind="stable")
                tp = float(yy[order[:k]].sum())
                lo = hi = tp
            rows.append({"model": name, "butce": f"%{int(b * 100)}",
                         "alarm": k, "kirli": int(yy.sum()),
                         "yakalanan": round(tp, 2),
                         "klor_esitlik_aralik": (f"{lo:.0f}-{hi:.0f}"
                                                 if name == "Egitimsiz klor"
                                                 else ""),
                         "kesinlik": tp / k, "duyarlilik": tp / yy.sum()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset()).reset_index(drop=True)
    y = df["y"].values
    routine = ~df["is_investigation"].values
    cl = df[CHLORINE]
    scores = {"Egitimsiz klor": -cl.fillna(cl.median()).values}
    for k, v in load_probs().items():
        scores[k] = v.reindex(range(len(df))).values
    rng = np.random.default_rng(20261112)

    # 1) Alarm butcesi
    bt = []
    for sname, mask in (("Rutin", routine), ("Tumu", np.ones(len(df), bool))):
        bt.append(budget_table(y, scores, mask, rng).assign(test=sname))
    bt = pd.concat(bt)
    bt.round(3).to_csv(OUT / "alarm_butcesi.csv", index=False)
    print(bt.round(3).to_string(index=False))

    # 2) Esik tablosu ve egrileri
    rows = []
    for name in ("A + gecmis + TabPFN", "A + TabPFN"):
        if name not in scores:
            continue
        for sname, mask in (("Tumu", np.ones(len(df), bool)),
                            ("Rutin", routine)):
            p, yy = scores[name][mask], y[mask]
            for t in THRESHOLDS:
                a = p >= t
                tp = int((a & (yy == 1)).sum())
                rows.append({"model": name, "test": sname, "esik": t,
                             "alarm": int(a.sum()), "yakalanan": tp,
                             "kirli": int(yy.sum()),
                             "kesinlik": tp / a.sum() if a.sum() else np.nan,
                             "duyarlilik": tp / yy.sum()})
    th = pd.DataFrame(rows)
    th.round(3).to_csv(OUT / "esik_tablosu.csv", index=False)
    print(th.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for name in ("Egitimsiz klor", "A + LR (ana model)", "A + gecmis + LR",
                 "A + gecmis + TabPFN"):
        if name in scores:
            pr, rc, _ = precision_recall_curve(y, scores[name])
            ax.plot(rc, pr, label=name.replace("gecmis", "geçmiş")
                    .replace("Egitimsiz", "Eğitimsiz"))
    ax.axhline(y.mean(), color="grey", ls=":", lw=1)
    ax.set_xlabel("Duyarlılık (yakalanan kirli / tüm kirli)")
    ax.set_ylabel("Kesinlik (doğru alarm / tüm alarm)")
    ax.set_title("Kesinlik–duyarlılık eğrileri (tüm numuneler)")
    ax.legend(fontsize=8)
    ax = axes[1]
    p = scores["A + gecmis + TabPFN"]
    grid = np.linspace(0.02, 0.8, 79)
    prec = [y[p >= t].mean() if (p >= t).any() else np.nan for t in grid]
    rec = [(y[p >= t]).sum() / y.sum() for t in grid]
    ax.plot(grid, prec, label="Kesinlik")
    ax.plot(grid, rec, label="Duyarlılık")
    ax.plot(grid, [(p >= t).mean() for t in grid], ls="--",
            label="Alarm verilen numune oranı")
    ax.set_xlabel("Alarm eşiği (TabPFN olasılığı)")
    ax.set_title("A + geçmiş + TabPFN: eşiğe göre")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "esik_egrileri.png", dpi=150)

    # 3) Basari egrisi
    df["hist_bin"] = pd.cut(df["HIST_N"], BINS, labels=BIN_LABELS)
    curve = []
    for b in BIN_LABELS:
        idx = np.where((df["hist_bin"] == b).values)[0]
        if y[idx].sum() in (0, len(idx)):
            continue
        row = {"bilinen_gecmis": b, "n": len(idx), "n_pos": int(y[idx].sum())}
        for name, s in scores.items():
            row[name] = roc_auc_score(y[idx], s[idx])
        curve.append(row)
    curve = pd.DataFrame(curve)
    curve.round(3).to_csv(OUT / "basari_egrisi_ROC.csv", index=False)
    print(curve.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r.bilinen_gecmis}\n({r.n}/{r.n_pos})"
              for r in curve.itertuples()]
    for name in ("Egitimsiz klor", "A + LR (ana model)", "A + gecmis + LR",
                 "A + gecmis + TabPFN", "GRU (secimli)"):
        if name in curve:
            ax.plot(labels, curve[name], "o-",
                    label=name.replace("gecmis", "geçmiş")
                    .replace("Egitimsiz", "Eğitimsiz")
                    .replace("secimli", "seçimli"),
                    lw=2.2 if name == "Egitimsiz klor" else 1.5)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Bölgede sonucu bilinen geçmiş numune sayısı "
                  "(numune/kirli)")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "basari_egrisi.png", dpi=150)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
