"""
Tech4Aqua - Bolge gecmisi (warm-start) ozellikleri.

Fikir: Yeni bir ZAPE izlenmeye basladiginda ilk numunelerin sonuclari
birkac gun icinde gelir. Model yeniden egitilmeden, bolgenin kendi
gecmis olcumlerini ozellik olarak kullanarak uyum saglayabilir mi?

Sizinti kurali (karar aninda bilinebilirlik):
  Bir numune icin yalnizca AYNI ZAPE'de, numune gununden en az
  LAB_DELAY_DAYS gun ONCE alinmis numuneler kullanilir. Koliform sonucu
  laboratuvardan 24-48 saatte geldigi icin daha yakin numunelerin
  sonucu karar aninda bilinmez. Ayni gunun numuneleri hicbir zaman
  kullanilmaz.

Ozellikler (gecmis yoksa NaN; model icinde medyanla doldurulur, ayrica
HIST_N = 0 bunu modele bildirir):
  HIST_N             : bilinen gecmis numune sayisi
  HIST_POS_RATE      : yumusatilmis gecmis pozitif orani
                       (pozitif + 1 * 0.1) / (n + 1)
  HIST_LAST_POS      : bilinen son numune pozitif miydi (0/1)
  HIST_DAYS_LAST     : bilinen son numuneden bu yana gecen gun (en fazla 730)
  HIST_DAYS_LAST_POS : bilinen son pozitiften bu yana gecen gun (en fazla 730)
  HIST_CL_MEAN       : gecmis numunelerin ortalama serbest klor degeri
"""

import numpy as np
import pandas as pd

from src.data import CHLORINE

LAB_DELAY_DAYS = 2
PRIOR_RATE, PRIOR_WEIGHT = 0.1, 1.0
CAP_DAYS = 730

HISTORY_FEATURES = ["HIST_N", "HIST_POS_RATE", "HIST_LAST_POS",
                    "HIST_DAYS_LAST", "HIST_DAYS_LAST_POS", "HIST_CL_MEAN"]


def add_history_features(df, lab_delay_days=LAB_DELAY_DAYS,
                         source_mask=None, prefix=None):
    """source_mask: verilirse gecmis yalnizca bu satirlardan hesaplanir
    (or. yalniz rutin numuneler). prefix: verilirse sutun adlari
    HIST_ yerine bu onekle yazilir (or. "HISTR_")."""
    if source_mask is not None:
        src_all = pd.Series(np.asarray(source_mask, bool), index=df.index)
    out = df.copy()
    for c in HISTORY_FEATURES:
        out[c] = np.nan
    for _, g in df.groupby("zape"):
        g = g.sort_values("date")
        dates = g["date"].values
        ys = g["y"].values
        cls = g[CHLORINE].values
        src = (src_all.loc[g.index].values if source_mask is not None
               else np.ones(len(g), bool))
        for idx, d in zip(g.index, dates):
            known = (dates <= d - np.timedelta64(lab_delay_days, "D")) & src
            n = int(known.sum())
            out.at[idx, "HIST_N"] = n
            if n == 0:
                continue
            ky, kd = ys[known], dates[known]
            out.at[idx, "HIST_POS_RATE"] = ((ky.sum() + PRIOR_WEIGHT *
                                            PRIOR_RATE) / (n + PRIOR_WEIGHT))
            last = np.argmax(kd)
            out.at[idx, "HIST_LAST_POS"] = ky[last]
            out.at[idx, "HIST_DAYS_LAST"] = min(
                (d - kd[last]) / np.timedelta64(1, "D"), CAP_DAYS)
            if ky.sum() > 0:
                lp = kd[ky == 1].max()
                out.at[idx, "HIST_DAYS_LAST_POS"] = min(
                    (d - lp) / np.timedelta64(1, "D"), CAP_DAYS)
            else:
                out.at[idx, "HIST_DAYS_LAST_POS"] = CAP_DAYS
            kc = cls[known]
            if np.isfinite(kc).any():
                out.at[idx, "HIST_CL_MEAN"] = np.nanmean(kc)
    if prefix is not None:
        new = {c: c.replace("HIST_", prefix) for c in HISTORY_FEATURES}
        out = out.rename(columns=new)
        out = df.join(out[list(new.values())])
    return out
