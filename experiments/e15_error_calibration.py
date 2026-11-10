"""
Deney 15 - Hata analizi ve kalibrasyon.

E13'un ZAPE gruplu CV test-disi olasiliklari (results/e13_tabpfn_history/
oof.csv) kullanilir; yeni model egitilmez (TabPFN gerekmez).

1) Hata analizi: her numune icin 3 tohumun ortalama olasiligi ve cogunluk
   karari alinir. Kacirilan kirlilikler (FN) ve yanlis alarmlar (FP)
   numune turu, belediye, klor duzeyi, bolge gecmisi uzunlugu ve mevsime
   gore sayilir. En emin yanlis tahminler listelenir.
2) Kalibrasyon: "model %30 diyorsa gercekten ~%30 mu?" sorusu.
   Esit sayili 5 olasilik kutusunda tahmin ortalamasi ile gercek oran,
   Brier skoru ve ECE (beklenen kalibrasyon hatasi) hesaplanir.
   Ayrica ZAPE gruplu capraz-fit Platt duzeltmesi denenir: bir ZAPE grubunun
   olasiliklari, diger gruplarin (olasilik, etiket) ciftleriyle egitilen
   lojistik bir egriyle yeniden olceklenir. Siralama (ROC/PR) degismez,
   yalnizca olasiliklarin dogrulugu degisir.

Calistirma: python -m experiments.e15_error_calibration
Ciktilar: results/e15_error_calibration/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.config import RESULTS_DIR
from src.data import CHLORINE, load_dataset
from src.history import add_history_features

OUT = RESULTS_DIR / "e15_error_calibration"
OOF = RESULTS_DIR / "e13_tabpfn_history" / "oof.csv"
MODELS = [("LR", "A"), ("LR", "A+HIST"), ("TABPFN", "A"),
          ("TABPFN", "A+HIST")]
N_BINS = 5


def ece(y, p, n_bins=N_BINS):
    """Esit sayili kutularla beklenen kalibrasyon hatasi."""
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    e = 0.0
    for b in np.unique(bins):
        m = bins == b
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def reliability(y, p, n_bins=N_BINS):
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(
        tahmin_ort=("p", "mean"), gercek_oran=("y", "mean"),
        n=("y", "size")).reset_index()


def platt_crossfit(df, p, n_splits=5):
    """ZAPE gruplu capraz-fit Platt duzeltmesi."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    out = np.zeros_like(p)
    y = df["y"].values
    for tr, te in GroupKFold(n_splits).split(z, y, df["zape"].values):
        lr = LogisticRegression(C=1e6).fit(z[tr, None], y[tr])
        out[te] = lr.predict_proba(z[te, None])[:, 1]
    return out


def bands(df):
    cl = df[CHLORINE]
    return {
        "Numune turu": np.where(df["is_investigation"], "Inceleme", "Rutin"),
        "Belediye": df["municipality"].astype(str).values,
        "Klor": np.select([cl.isna(), cl < 0.2, cl < 0.5],
                          ["Olculmemis", "<0.2", "0.2-0.5"], ">=0.5"),
        "Bolge gecmisi (onceki numune)": pd.cut(
            df["HIST_N"], [-1, 0, 2, 5, 10, 1e9],
            labels=["0", "1-2", "3-5", "6-10", "11+"]).astype(str),
        "Mevsim": pd.Series(df["date"].dt.month).map(
            lambda m: "Kis" if m in (12, 1, 2) else "Ilkbahar"
            if m in (3, 4, 5) else "Yaz" if m in (6, 7, 8) else "Sonbahar"
        ).values,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_history_features(load_dataset()).reset_index(drop=True)
    y = df["y"].values
    oof = pd.read_csv(OOF)
    oof = oof[oof["protocol"] == "zape_cv"]
    groups = bands(df)

    err_rows, cal_rows, rel_rows, top_rows = [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for model, fs in MODELS:
        g = oof[(oof["model"] == model) & (oof["feature_set"] == fs)]
        agg = g.groupby("row").agg(p=("prob", "mean"),
                                   pred=("pred", "mean")).sort_index()
        p = agg["p"].values
        pred = (agg["pred"].values >= 0.5).astype(int)
        name = f"{model} {fs}"

        # --- hata analizi
        for gname, gval in groups.items():
            for v in pd.unique(gval):
                m = gval == v
                err_rows.append({
                    "model": name, "grup": gname, "deger": v,
                    "n": int(m.sum()), "kirli": int(y[m].sum()),
                    "yakalanan_TP": int(((pred == 1) & (y == 1) & m).sum()),
                    "kacirilan_FN": int(((pred == 0) & (y == 1) & m).sum()),
                    "yanlis_alarm_FP": int(((pred == 1) & (y == 0)
                                            & m).sum()),
                    "temiz": int((y[m] == 0).sum()),
                })
        if (model, fs) == ("TABPFN", "A+HIST"):
            d = df.assign(p=p, pred=pred)
            cols = ["date", "municipality", "zape", "is_investigation",
                    CHLORINE, "HIST_N", "HIST_POS_RATE", "y", "p"]
            top_rows.append(d[(y == 1) & (pred == 0)]
                            .nsmallest(10, "p")[cols]
                            .assign(tur="kacirilan (en dusuk olasilik)"))
            top_rows.append(d[(y == 0) & (pred == 1)]
                            .nlargest(10, "p")[cols]
                            .assign(tur="yanlis alarm (en yuksek olasilik)"))

        # --- kalibrasyon
        pc = platt_crossfit(df, p)
        for label, pp in (("ham", p), ("Platt duzeltmeli", pc)):
            cal_rows.append({
                "model": name, "olasilik": label,
                "ortalama_tahmin": pp.mean(), "gercek_oran": y.mean(),
                "brier": np.mean((pp - y) ** 2), "ECE": ece(y, pp)})
            r = reliability(y, pp).assign(model=name, olasilik=label)
            rel_rows.append(r)
            ax = axes[0 if label == "ham" else 1]
            ax.plot(r["tahmin_ort"], r["gercek_oran"], "o-", label=name)
    for ax, t in zip(axes, ("Ham olasılıklar", "Platt düzeltmeli")):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Mükemmel kalibrasyon")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(t)
        ax.set_xlabel("Modelin verdiği olasılık (kutu ortalaması)")
    axes[0].set_ylabel("Gerçek kirlilik oranı")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kalibrasyon_egrisi.png", dpi=150)

    err = pd.DataFrame(err_rows)
    err["kacirma_orani"] = err["kacirilan_FN"] / err["kirli"].replace(0, np.nan)
    err["yanlis_alarm_orani"] = (err["yanlis_alarm_FP"]
                                 / err["temiz"].replace(0, np.nan))
    err.round(3).to_csv(OUT / "hata_dagilimi.csv", index=False)
    cal = pd.DataFrame(cal_rows).round(4)
    cal.to_csv(OUT / "kalibrasyon.csv", index=False)
    pd.concat(rel_rows).round(4).to_csv(OUT / "guvenilirlik_kutulari.csv",
                                        index=False)
    pd.concat(top_rows).to_csv(OUT / "en_emin_hatalar_tabpfn.csv",
                                        index=False)
    print(cal.to_string(index=False))
    sel = err[err["model"].isin(["LR A", "TABPFN A+HIST"])]
    print(sel.round(2).to_string(index=False))
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
