import sys; sys.path.insert(0, "/home/claude/tech4aqua")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd
from src.data import load_dataset, CHLORINE
OUT = "/home/claude/figs_tb/"
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET = "#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

# ---------- Sekil 1: on calisma
models = ["Eğitimsiz klor", "Lojistik Regresyon", "1D-CNN (tablo)", "TabPFN",
          "LR + bölge geçmişi", "GRU (dizi, seçimli)", "TabPFN + bölge geçmişi"]
roc_t = [0.720, 0.754, 0.686, 0.801, 0.825, 0.822, 0.878]
roc_r = [0.848, 0.782, 0.731, 0.752, 0.748, 0.684, 0.814]
pr_t  = [0.281, 0.347, 0.276, 0.422, 0.519, 0.493, 0.563]
pr_r  = [0.270, 0.196, 0.177, 0.132, 0.380, 0.266, 0.328]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
y = np.arange(len(models))[::-1]; h = 0.36
for ax, (t, r, ttl, base) in zip(axes, [(roc_t, roc_r, "(a) ROC-AUC", (0.5, 0.5)),
                                         (pr_t, pr_r, "(b) PR-AUC", (0.157, 0.064))]):
    ax.barh(y + h/2 + 0.01, t, h, color=BLUE, label="Tüm numuneler (522; 82 uygunsuz)")
    ax.barh(y - h/2 - 0.01, r, h, color=ORANGE, label="Rutin numuneler (390; 25 uygunsuz)")
    for yy, a, b in zip(y, t, r):
        ax.text(a + 0.01, yy + h/2, f"{a:.2f}".replace(".", ","), va="center", fontsize=7.5, color=INK2)
        ax.text(b + 0.01, yy - h/2, f"{b:.2f}".replace(".", ","), va="center", fontsize=7.5, color=INK2)
    ax.axvline(base[0], color=BLUE, ls=":", lw=1); 
    if base[1] != base[0]: ax.axvline(base[1], color=ORANGE, ls=":", lw=1)
    ax.set_title(ttl + "  (noktalı çizgi: rastgele düzey)", loc="left", fontsize=9.5, color=INK); ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_yticks(y); axes[0].set_yticklabels(models)
axes[0].legend(loc="lower center", bbox_to_anchor=(1.0, -0.32), ncol=2, frameon=False, fontsize=8)
fig.subplots_adjust(left=0.2, right=0.98, wspace=0.08, bottom=0.2); fig.savefig(OUT + "s1_on_calisma.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Sekil 2: is akisi
fig, ax = plt.subplots(figsize=(9.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
def box(x, y, w, hh, txt, ip, color, dashed=False):
    p = FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.3,rounding_size=1.2", fc=color + "22",
                       ec=color, lw=1.4, ls="--" if dashed else "-")
    ax.add_patch(p); ax.text(x + w/2, y + hh/2 + 1.6, txt, ha="center", va="center", fontsize=7.8, color=INK)
    ax.text(x + w/2, y + 1.6, ip, ha="center", va="center", fontsize=7.5, color=color, weight="bold")
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
box(1, 21, 17, 12, "Açık veri\n(Tech4Aqua,\nOpen-Meteo)", "", INK2)
box(22, 21, 17, 12, "Etiket düzeltme,\nsızıntı denetimi,\nözellikler", "İP1", BLUE)
box(43, 21, 17, 12, "3 test senaryosu\n(yeni bölge, gelecek\nyıl, yeni belediye)", "İP1", BLUE)
box(64, 21, 16, 12, "Tablo, önceden\neğitilmiş ve derin\nmodeller", "İP2", ORANGE)
box(83, 21, 16, 12, "Bölge geçmişi\nve dizi modelleri", "İP3", AQUA)
box(64, 3, 16, 12, "Bootstrap, kalibrasyon,\nalarm bütçesi,\nhata analizi", "İP4", VIOLET)
box(43, 3, 17, 12, "Nihai model\nseçimi (klora\nkarşı kazanç)", "İP4", VIOLET)
box(22, 3, 17, 12, "Erken uyarı\ndemo uygulaması", "İP5", GREEN)
box(1, 3, 17, 12, "Rapor, açık kod\nve bildiri", "İP5", GREEN, dashed=True)
for a, b in [((18.5, 27), (21.5, 27)), ((39.5, 27), (42.5, 27)), ((60.5, 27), (63.5, 27)), ((80.5, 27), (82.5, 27)),
             ((91, 20.5), (78, 15.5)), ((72, 20.5), (72, 15.5)), ((63.5, 9), (60.5, 9)), ((42.5, 9), (39.5, 9)), ((21.5, 9), (18.5, 9))]:
    arrow(*a, *b)
fig.savefig(OUT + "s2_is_akisi.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Sekil 3: veri
df = load_dataset()
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax = axes[0]
yr = df.groupby(["year", "is_investigation"]).size().unstack(fill_value=0).reindex(range(2008, 2026), fill_value=0)
pos = df.groupby("year")["y"].sum().reindex(range(2008, 2026), fill_value=0)
ax.bar(yr.index, yr[False], color=BLUE, label="Rutin", width=0.8)
ax.bar(yr.index, yr[True], bottom=yr[False], color=ORANGE, label="Soruşturma", width=0.8)
ax.plot(pos.index, pos.values, "o", color=INK, ms=3.5, label="Uygunsuz")
ax.set_title("(a) Yıllara göre numune sayısı", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xticks(range(2008, 2026, 3))
ax = axes[1]
cl = df[CHLORINE]
for k, (lab, col) in enumerate([("Uygun", BLUE), ("Uygunsuz", ORANGE)]):
    v = cl[df["y"] == k].dropna().clip(upper=1.5)
    ax.boxplot(v, positions=[k], widths=0.5, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor=col + "55", edgecolor=col), medianprops=dict(color=INK),
               whiskerprops=dict(color=col), capprops=dict(color=col),
               flierprops=dict(marker="o", ms=2.5, mfc=col, mec=col, alpha=.5))
ax.set_xticks([0, 1]); ax.set_xticklabels(["Uygun\n(n=440)", "Uygunsuz\n(n=82)"])
ax.set_ylabel("Serbest klor (mg/L, 1,5'te kesildi)"); ax.set_title("(b) Klor ve sonuç", loc="left", fontsize=9.5)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax = axes[2]
cnt = df.groupby("zape").size().sort_values(ascending=False)
ax.bar(range(len(cnt)), cnt.values, color=AQUA, width=0.9)
ax.set_yscale("log"); ax.set_xlabel("ZAPE (çoktan aza sıralı)"); ax.set_ylabel("Numune sayısı (log)")
ax.set_title("(c) ZAPE başına numune", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT + "s3_veri.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("n uygun", (df.y==0).sum(), "uygunsuz", df.y.sum(), "en buyuk zape", cnt.iloc[0], "tek numuneli zape", (cnt==1).sum())

# ---------- Sekil 4: basari egrisi
c = pd.read_csv("/home/claude/tech4aqua/results/e17_decision_analysis/basari_egrisi_ROC.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.4))
lab = [f"{a}\n({n}/{p})" for a, n, p in zip(c.bilinen_gecmis, c.n, c.n_pos)]
for col, name, color, lw in [("Egitimsiz klor", "Eğitimsiz klor", INK2, 2.4), ("A + LR (ana model)", "Lojistik Regresyon", BLUE, 1.6),
                             ("A + gecmis + LR", "LR + bölge geçmişi", AQUA, 1.6), ("GRU (secimli)", "GRU (dizi, seçimli)", VIOLET, 1.6),
                             ("A + gecmis + TabPFN", "TabPFN + bölge geçmişi", ORANGE, 2.0)]:
    if col in c: ax.plot(lab, c[col], "o-", color=color, lw=lw, ms=5, label=name)
ax.axhline(0.5, color=GRID, lw=1); ax.set_ylim(0.3, 1.0); ax.set_ylabel("ROC-AUC")
ax.set_xlabel("Bölgede sonucu bilinen önceki numune sayısı (numune/uygunsuz)")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
fig.tight_layout(); fig.savefig(OUT + "s4_basari_egrisi.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(c.round(3).to_string())
