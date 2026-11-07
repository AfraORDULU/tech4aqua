"""
Tech4Aqua - Veri seti ve veri analizi raporu (1. danisman toplantisi).

Danismanin istedigi 5 baslik icin tablo ve grafik uretir:
  1. Veri setinin temini ve mevcut durumu    -> 01_dataset_overview.csv
  2. Genel ozellikler ve dagilimlar          -> 02_*.csv, fig_01..fig_03
  3. Etiketleme sureci ve etiket kontrolu     -> 03_label_audit.csv, label_changes.csv
  4. Veri kalitesi, eksik veri, on isleme     -> 04_*.csv, fig_04, fig_05
  5. (Tamamlananlar / sonrakiler raporda yazilir)

Calistirma (proje kok klasorunden, once build_dataset):
    python -m src.build_dataset
    python -m src.data_quality_report

Ciktilar: results/data_audit/  ve  results/data_audit/figures/
Bu betik yalnizca veriyi betimler; model egitmez.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_CSV,
    FEATURES_CSV,
    RAW_XLSX,
    DATA_DIR,
    DATA_AUDIT_DIR,
)
from src.preprocessing import load_and_preprocess_data

TARGET = "Bactérias coliformes"
OUT = DATA_AUDIT_DIR
FIG = DATA_AUDIT_DIR / "figures"

# Dogrulanmis kategorik palet (dataviz referans paleti, slot 1-2)
C_NEG = "#2a78d6"   # negatif / uygun
C_POS = "#eb6834"   # pozitif / uygunsuz
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

METEO = ["TEMP MÁX (ºC)", "PRECIPITAÇÃO (mm)", "UMIDADE MÉDIA (%)"]


# ------------------------------------------------------------------
# Yardimcilar
# ------------------------------------------------------------------

def rate_table(df, by):
    t = df.groupby(by, dropna=False)["y"].agg(
        Numune="size", Pozitif="sum"
    )
    t["Negatif"] = t["Numune"] - t["Pozitif"]
    t["Pozitif_Orani"] = (t["Pozitif"] / t["Numune"]).round(3)
    return t.reset_index()


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# Veri
# ------------------------------------------------------------------

def load():
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            "Islenmis veri yok. Once: python -m src.build_dataset"
        )
    full = pd.read_csv(PROCESSED_CSV)
    df = full[full[TARGET].notna()].copy()
    df["y"] = (df[TARGET] > 0).astype(int)
    df["date"] = pd.to_datetime(df["DATA AMOSTRAGEM"])
    df["Yil"] = df["date"].dt.year
    df["Belediye"] = df["FREGUESIA"]
    df["Ornekleme"] = df["META_SAMPLING_TYPE"]
    return full, df


# ------------------------------------------------------------------
# 1. Genel durum
# ------------------------------------------------------------------

def overview(full, df, X):
    raw = pd.read_excel(RAW_XLSX, sheet_name="MEDIÇÕES INDIVIDUAIS")
    n_coli_raw = int((raw["PARAMETRO NOME"] == TARGET).sum())
    zape_counts = df["ZAPE NOME"].value_counts()

    rows = [
        ("Ham kayit sayisi (tum parametreler)", len(raw)),
        ("Ham parametre sayisi", raw["PARAMETRO NOME"].nunique()),
        ("Numune olayi (ozellik tablosu satiri)", len(full)),
        ("Ozellik tablosu sutun sayisi (orijinal)",
         len([c for c in full.columns if not c.startswith("META_")])),
        ("Ham koliform kaydi", n_coli_raw),
        ("Etiketli numune (modelleme satiri)", len(df)),
        ("Ayni ZAPE+gun birlestirilen fazla kayit",
         int((df["META_N_COLIFORM_RECORDS"] - 1).sum())),
        ("Pozitif (koliform > 0)", int(df["y"].sum())),
        ("Negatif", int((1 - df["y"]).sum())),
        ("Pozitif orani", round(df["y"].mean(), 3)),
        ("Duzeltilen etiket ('<' operatoru)",
         int(df["META_LABEL_CHANGED"].sum())),
        ("Etiketli ZAPE sayisi", df["ZAPE NOME"].nunique()),
        ("Hic pozitifi olmayan ZAPE",
         int((df.groupby("ZAPE NOME")["y"].sum() == 0).sum())),
        ("Tek numunesi olan ZAPE", int((zape_counts == 1).sum())),
        ("En buyuk 3 ZAPE'nin numune payi",
         round(zape_counts.head(3).sum() / len(df), 3)),
        ("Belediye sayisi", df["Belediye"].nunique()),
        ("Etiketli ilk tarih", df["date"].min().date()),
        ("Etiketli son tarih", df["date"].max().date()),
        ("Modelde kullanilan ozellik sayisi", X.shape[1]),
        ("Sorusturma numunesi sayisi", int(df["META_IS_INVESTIGATION"].sum())),
    ]
    t = pd.DataFrame(rows, columns=["Bilgi", "Deger"])
    t.to_csv(OUT / "01_dataset_overview.csv", index=False)
    return t


# ------------------------------------------------------------------
# 2. Dagilimlar
# ------------------------------------------------------------------

def distributions(df, X):
    rate_table(df, "Belediye").to_csv(
        OUT / "02a_class_by_municipality.csv", index=False)
    rate_table(df, "Yil").to_csv(
        OUT / "02b_class_by_year.csv", index=False)
    rate_table(df, "Ornekleme").to_csv(
        OUT / "02c_class_by_sampling_type.csv", index=False)
    rate_table(df, ["Yil", "Ornekleme"]).to_csv(
        OUT / "02d_class_by_year_and_sampling.csv", index=False)
    rate_table(df, "META_LAB").rename(
        columns={"META_LAB": "Laboratuvar"}).to_csv(
        OUT / "02e_class_by_lab.csv", index=False)

    z = rate_table(df, ["Belediye", "ZAPE NOME"]).sort_values(
        "Numune", ascending=False)
    z.to_csv(OUT / "02f_zape_summary.csv", index=False)

    # Model ozelliklerinin sinifa gore ozeti (medyan ve ceyrekler)
    rows = []
    for col in X.columns:
        for label, part in df.groupby("y"):
            s = part[col].dropna()
            rows.append({
                "Ozellik": col,
                "Sinif": "Pozitif" if label == 1 else "Negatif",
                "N": len(s),
                "Medyan": s.median(),
                "Q1": s.quantile(0.25),
                "Q3": s.quantile(0.75),
                "Ortalama": s.mean(),
            })
    pd.DataFrame(rows).round(4).to_csv(
        OUT / "02g_feature_summary_by_class.csv", index=False)

    # ---- Grafik 1: yila gore numune sayisi (ornekleme turune gore) ve pozitif orani
    years = sorted(df["Yil"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    bottom = np.zeros(len(years))
    parts = [
        ("Rutin", 0, C_NEG, 1.0, "Rutin - negatif"),
        ("Rutin", 1, C_POS, 1.0, "Rutin - pozitif"),
        ("Sorusturma", 0, C_NEG, 0.4, "Soruşturma - negatif"),
        ("Sorusturma", 1, C_POS, 0.4, "Soruşturma - pozitif"),
    ]
    for kind, label, color, alpha, name in parts:
        vals = np.array([
            ((df["Yil"] == yr) & (df["Ornekleme"] == kind)
             & (df["y"] == label)).sum() for yr in years
        ])
        ax.bar(years, vals, bottom=bottom, color=color, alpha=alpha,
               width=0.7, edgecolor="white", linewidth=1, label=name)
        bottom += vals
    ax.set_ylabel("Numune sayısı", color=INK_2)
    ax.set_title("Yıllara göre etiketli numuneler (soluk = soruşturma numunesi)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    style_axes(ax)

    ax = axes[1]
    yr_rate = df.groupby("Yil")["y"].mean()
    rt = df[df["Ornekleme"] == "Rutin"].groupby("Yil")["y"].mean()
    ax.plot(yr_rate.index, yr_rate.values, color=C_POS, linewidth=2,
            marker="o", markersize=5, label="Tüm numuneler")
    ax.plot(rt.index, rt.values, color=C_NEG, linewidth=2,
            marker="o", markersize=5, label="Yalnız rutin numuneler")
    ax.set_ylabel("Pozitif oranı", color=INK_2)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    last = max(years)
    last_df = df[df["Yil"] == last]
    note = (f"{last}: {len(last_df)} numunenin {int(last_df['y'].sum())}'i pozitif")
    if (last_df["Ornekleme"] == "Sorusturma").all():
        note += "\n(tamamı soruşturma numunesi)"
    ax.annotate(note,
                xy=(last, yr_rate.loc[last]), xytext=(last - 5.7, 0.75),
                fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    style_axes(ax)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    save(fig, "fig_01_year_sampling.png")

    # ---- Grafik 2: belediye ve ornekleme turune gore pozitif orani
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, col, title in [
        (axes[0], "Belediye", "Belediyeye göre"),
        (axes[1], "Ornekleme", "Örnekleme türüne göre"),
    ]:
        t = rate_table(df, col)
        labels = [
            str(v).replace("Sorusturma", "Soruşturma") for v in t[col]
        ]
        ax.bar(labels, t["Pozitif_Orani"], color=C_POS, width=0.55)
        for i, r in t.iterrows():
            ax.text(i, r["Pozitif_Orani"] + 0.015,
                    f"{int(r['Pozitif'])}/{int(r['Numune'])}",
                    ha="center", fontsize=8, color=INK_2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("Pozitif oranı", color=INK_2)
    save(fig, "fig_02_rate_by_group.png")

    # ---- Grafik 3: ZAPE basina numune sayisi
    counts = df["ZAPE NOME"].value_counts().values
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(1, len(counts) + 1), counts, color=C_NEG, width=0.8)
    ax.set_xlabel("ZAPE (numune sayısına göre sıralı)", color=INK_2)
    ax.set_ylabel("Numune sayısı", color=INK_2)
    ax.set_title(
        f"{len(counts)} ZAPE: ilk 3 ZAPE numunelerin "
        f"%{100 * counts[:3].sum() / counts.sum():.0f}'ini içeriyor, "
        f"{(counts == 1).sum()} ZAPE'nin tek numunesi var",
        color=INK, fontsize=10, loc="left")
    style_axes(ax)
    save(fig, "fig_03_zape_sizes.png")


# ------------------------------------------------------------------
# 3. Etiket kontrolu
# ------------------------------------------------------------------

def label_audit(df):
    raw = pd.read_excel(RAW_XLSX, sheet_name="MEDIÇÕES INDIVIDUAIS")
    coli = raw[raw["PARAMETRO NOME"] == TARGET]
    ops = coli["OPERADOR"].astype(str).str.strip().value_counts()

    rows = [
        ("Etiket kaynagi",
         "Laboratuvar koliform sayimi (N/100 mL); elle etiketleme yok"),
        ("Etiket kurali", "Koliform > 0 ise 1 (DL 306/2007, 0/100 mL siniri)"),
        ("Ham koliform kaydi", len(coli)),
        ("Birim(ler)", " | ".join(coli["UNIDADES"].astype(str).str.strip().unique())),
        ("Operator '=' kaydi", int(ops.get("=", 0))),
        ("Operator '>' kaydi (ust olcum siniri, pozitif)", int(ops.get(">", 0))),
        ("Operator '<' kaydi (tespit siniri alti)", int(ops.get("<", 0))),
        ("Ayni ZAPE+gunde birden fazla kayit olan numune",
         int((df["META_N_COLIFORM_RECORDS"] > 1).sum())),
        ("Birlestirme kurali", "Ayni gun kayitlarinin ortalamasi > 0 ise pozitif"),
        ("Yazarlarin etiketi ham veriden yeniden uretildi mi", "Evet (522/522)"),
        ("'<' nedeniyle yanlis pozitif olup duzeltilen", int(df["META_LABEL_CHANGED"].sum())),
        ("Supheli kesirli sonuc (0<x<1, '=')", int(df["META_FRACTIONAL_RESULT"].sum())),
        ("Kesirli sonuclu numunelerin ornekleme turu",
         " | ".join(df.loc[df["META_FRACTIONAL_RESULT"] == 1, "Ornekleme"].unique())),
    ]
    t = pd.DataFrame(rows, columns=["Kontrol", "Sonuc"])
    t.to_csv(OUT / "03_label_audit.csv", index=False)

    df.loc[df["META_FRACTIONAL_RESULT"] == 1,
           ["Belediye", "ZAPE NOME", "DATA AMOSTRAGEM", TARGET,
            "META_CONTROL_TYPE", "META_SOURCE_FILE"]].to_csv(
        OUT / "03b_fractional_results.csv", index=False)
    return t


# ------------------------------------------------------------------
# 4. Veri kalitesi
# ------------------------------------------------------------------

def missingness(df, X):
    rows = []
    for col in X.columns:
        miss = df[col].isna()
        rows.append({
            "Ozellik": col,
            "Eksik": int(miss.sum()),
            "Eksik_Yuzde": round(100 * miss.mean(), 1),
            "Eksiklerde_Pozitif_Orani":
                round(df.loc[miss, "y"].mean(), 3) if miss.any() else np.nan,
            "Doluda_Pozitif_Orani": round(df.loc[~miss, "y"].mean(), 3),
        })
    t = pd.DataFrame(rows).sort_values("Eksik", ascending=False)
    t.to_csv(OUT / "04a_missingness.csv", index=False)

    # Grafik 4: klor dagilimi (sinifa gore) + eksik klor
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    bins = np.arange(0, 1.55, 0.05)
    for label, color, name in [(0, C_NEG, "Negatif"), (1, C_POS, "Pozitif")]:
        s = df.loc[df["y"] == label, "Desinfetante residual"].dropna()
        ax.hist(s.clip(upper=1.5), bins=bins, density=True, color=color,
                alpha=0.55 if label == 0 else 0.75, label=f"{name} (n={len(s)})")
    ax.axvline(0.2, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0.21, ax.get_ylim()[1] * 0.9, "0,2 mg/L\nyasal alt sınır",
            fontsize=8, color=INK_2)
    ax.set_xlabel("Kalıntı klor (mg/L, 1,5 üstü kırpıldı)", color=INK_2)
    ax.set_ylabel("Yoğunluk", color=INK_2)
    ax.set_title("Klor dağılımı sınıfa göre", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1]
    miss = df["META_CHLORINE_MISSING"] == 1
    vals = [df.loc[~miss, "y"].mean(), df.loc[miss, "y"].mean()]
    ax.bar(["Klor ölçülmüş", "Klor eksik"], vals, color=C_POS, width=0.55)
    for i, (m, n) in enumerate([(~miss, None), (miss, None)]):
        ax.text(i, vals[i] + 0.02,
                f"{int(df.loc[m, 'y'].sum())}/{int(m.sum())}",
                ha="center", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pozitif oranı", color=INK_2)
    ax.set_title("Eksik klor = çoğunlukla pozitif", color=INK,
                 fontsize=10, loc="left")
    style_axes(ax)
    save(fig, "fig_04_chlorine.png")

    # Grafik 5: modeldeki ozelliklerin eksik orani
    t2 = t[t["Eksik"] > 0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(t2["Ozellik"][::-1], t2["Eksik_Yuzde"][::-1], color=C_NEG,
            height=0.6)
    for i, v in enumerate(t2["Eksik_Yuzde"][::-1]):
        ax.text(v + 0.5, i, f"%{v}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Eksik oranı (%) - 522 etiketli numune", color=INK_2)
    ax.set_title("Model özelliklerinde eksik veri", color=INK,
                 fontsize=10, loc="left")
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID)
    ax.yaxis.grid(False)
    save(fig, "fig_05_missingness.png")
    return t


def clustering_and_meteo(df):
    """Ayni belediye + ayni gun numuneleri ve hava verisinin bolgeselligi."""
    key = df["Belediye"] + "_" + df["DATA AMOSTRAGEM"].astype(str)
    size = key.map(key.value_counts())
    multi = df[size > 1]

    multi_keys = key[size > 1].unique()

    # ayni gun pozitif kumelenmesi
    day = df.groupby(key)["y"].agg(["size", "sum"])
    day = day[day["size"] > 1]

    rows = [
        ("Ayni belediye+gunu baska etiketli numuneyle paylasan numune",
         len(multi)),
        ("  bunlarin pozitif olani", int(multi["y"].sum())),
        ("Birden fazla numune alinan belediye-gun sayisi", len(multi_keys)),
        ("  en az 2 pozitif iceren belediye-gun", int((day["sum"] >= 2).sum())),
        ("Etiketli numunelerde benzersiz tarih", df["DATA AMOSTRAGEM"].nunique()),
    ]

    # Hava verisi ne kadar "gune ozel"? Ayni belediye-gun icindeki fark ile
    # ayni belediyede rastgele iki farkli gun arasindaki farki karsilastir.
    rng = np.random.default_rng(0)
    for col in METEO:
        within = multi.groupby(key[size > 1])[col].agg(
            lambda s: s.max() - s.min())
        between = []
        for _, part in df.groupby("Belediye"):
            v = part[col].dropna().to_numpy()
            if len(v) > 1:
                i = rng.integers(0, len(v), 2000)
                j = rng.integers(0, len(v), 2000)
                between.append(np.abs(v[i] - v[j]))
        between = np.concatenate(between)
        rows.append((f"{col}: ayni gun birebir ayni olan belediye-gun orani",
                     round((within == 0).mean(), 3)))
        rows.append((f"{col}: ayni gun medyan fark",
                     round(float(within.median()), 2)))
        rows.append((f"{col}: farkli gunler medyan fark",
                     round(float(np.median(between)), 2)))
    t = pd.DataFrame(rows, columns=["Kontrol", "Sonuc"])
    t.to_csv(OUT / "04b_same_day_clustering.csv", index=False)
    return t


def feature_provenance(full):
    """
    Yazarlarin 01_feature_engineering.py betigini ham veride calistirip
    paylasilan ozellik tablosuyla sutun sutun karsilastirir.
    """
    script = DATA_DIR / "raw" / "02_Scripts" / "01_feature_engineering.py"
    if not script.exists():
        print("Uyari: yazar betigi bulunamadi, kaynak kontrolu atlandi.")
        return None

    spec = importlib.util.spec_from_file_location("authors_fe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "repro.csv"
        module.generate_engineered_features(str(RAW_XLSX), str(out))
        repro = pd.read_csv(out)

    orig = pd.read_csv(FEATURES_CSV)
    keys = ["ZAPE NOME", "DATA AMOSTRAGEM"]
    for d in (repro, orig):
        d["DATA AMOSTRAGEM"] = pd.to_datetime(
            d["DATA AMOSTRAGEM"]).dt.strftime("%Y-%m-%d")
    m = orig.merge(repro, on=keys, how="left", suffixes=("", "__r"))
    labeled = m[TARGET].notna()

    _, _, _, groups = load_and_preprocess_data(PROCESSED_CSV)
    model_cols = [c for g in groups.values() for c in g]

    rows = []
    for col in model_cols:
        if col + "__r" not in m.columns:
            status = "Uretim kodu pakette YOK"
            n_diff = np.nan
        else:
            a = m.loc[labeled, col]
            b = m.loc[labeled, col + "__r"]
            n_diff = int((~np.isclose(a, b, equal_nan=True)).sum())
            status = ("Ham veriden birebir uretildi" if n_diff == 0
                      else "Ham veriden uretildi, farklar var")
        rows.append({"Ozellik": col, "Durum": status,
                     "Farkli_Satir_(522_icinde)": n_diff})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "04c_feature_provenance.csv", index=False)
    return t


# ------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    full, df = load()
    X, y, _, _ = load_and_preprocess_data(PROCESSED_CSV)
    assert (X.index == df.index).all()

    pd.set_option("display.width", 200)
    print("\n=== 1. GENEL DURUM ===")
    print(overview(full, df, X).to_string(index=False))

    distributions(df, X)
    print("\n=== 2. DAGILIMLAR ===")
    for f in ["02a_class_by_municipality.csv", "02c_class_by_sampling_type.csv",
              "02e_class_by_lab.csv"]:
        print(pd.read_csv(OUT / f).to_string(index=False), "\n")

    print("=== 3. ETIKET KONTROLU ===")
    print(label_audit(df).to_string(index=False))

    print("\n=== 4. VERI KALITESI ===")
    print(missingness(df, X).head(10).to_string(index=False))
    print()
    print(clustering_and_meteo(df).to_string(index=False))
    prov = feature_provenance(full)
    if prov is not None:
        print()
        print(prov.to_string(index=False))

    print("\nCiktilar:", OUT)


if __name__ == "__main__":
    main()
