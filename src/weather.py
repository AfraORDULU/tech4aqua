"""
Tech4Aqua - Gunluk hava verisinden gecikmeli (antecedent) ozellikler.

Neden gerekli?
--------------
Veri setinde hava durumu yalnizca numune gunu icin var. Veri setindeki
PRECIP_LAG1_REGIONAL "bir onceki gun" degil, ayni bolgedeki bir onceki
NUMUNENIN gunudur (haftalar oncesi olabilir). Kontaminasyon genellikle
onceki birkac gunun yagisiyla iliskili oldugu icin gercek gunluk seri
gerekir.

Kaynak: Open-Meteo Historical Weather API (ERA5 tabanli), 3 belediye
merkezi icin gunluk yagis ve sicaklik, 2007-12-01 .. 2025-12-31.
Dosya: data/external/open_meteo_daily.csv (indirme baglantisi README'de).

Sizinti kurali: Tum ozellikler numune gunundan ONCEKI gunlerden
(D-1 ... D-k) hesaplanir. Numune gunu (D) dahil edilmez.
"""

import io

import numpy as np
import pandas as pd

from src.config import DATA_DIR

WEATHER_CSV = DATA_DIR / "external" / "open_meteo_daily.csv"

# Open-Meteo location_id -> belediye (indirme baglantisindaki sira)
LOCATIONS = {0: "CHAVES", 1: "MONTALEGRE", 2: "BOTICAS"}

LAG_FEATURES = ["RAIN_3D", "RAIN_7D", "RAIN_14D", "WET_DAYS_7D",
                "DAYS_SINCE_RAIN", "TMEAN_7D"]


def load_daily_weather(path=WEATHER_CSV):
    """Open-Meteo CSV'sini (iki bolumlu) okuyup uzun tabloya cevirir."""
    text = open(path, encoding="utf-8").read()
    # Dosyada once konum tablosu, bos satirdan sonra gunluk veri var.
    daily_part = text.split("\n\n", 1)[1]
    w = pd.read_csv(io.StringIO(daily_part))
    w.columns = ["location_id", "date", "rain", "tmax", "tmean"]
    w["municipality"] = w["location_id"].map(LOCATIONS)
    w["date"] = pd.to_datetime(w["date"])
    return w.sort_values(["municipality", "date"]).reset_index(drop=True)


def _features_for_town(w):
    w = w.set_index("date").asfreq("D")
    prev_rain = w["rain"].shift(1)        # D-1'den baslar, D dahil degil
    prev_t = w["tmean"].shift(1)
    out = pd.DataFrame(index=w.index)
    out["RAIN_3D"] = prev_rain.rolling(3).sum()
    out["RAIN_7D"] = prev_rain.rolling(7).sum()
    out["RAIN_14D"] = prev_rain.rolling(14).sum()
    out["WET_DAYS_7D"] = (prev_rain >= 1.0).rolling(7).sum()
    out["TMEAN_7D"] = prev_t.rolling(7).mean()
    # Son >= 1 mm yagisli gunden bu yana gecen gun (D-1 yagisliysa 1)
    wet = (w["rain"] >= 1.0).values
    days, last = np.full(len(w), np.nan), None
    for i in range(len(w)):
        if last is not None:
            days[i] = i - last
        if wet[i]:
            last = i
    out["DAYS_SINCE_RAIN"] = np.minimum(days, 60)
    return out


def add_lag_features(df, path=WEATHER_CSV):
    """df'ye (load_dataset ciktisi) gecikmeli hava ozelliklerini ekler."""
    w = load_daily_weather(path)
    feats = []
    for town, g in w.groupby("municipality"):
        f = _features_for_town(g)
        f["municipality"] = town
        feats.append(f.reset_index())
    feats = pd.concat(feats)
    out = df.merge(feats, on=["municipality", "date"], how="left",
                   validate="many_to_one")
    assert len(out) == len(df)
    missing = out[LAG_FEATURES].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} numune icin hava verisi bulunamadi")
    return out


def same_day_agreement(df, path=WEATHER_CSV):
    """Open-Meteo ile veri setindeki numune gunu havasi ne kadar uyumlu?"""
    w = load_daily_weather(path)
    m = df.merge(w, on=["municipality", "date"], how="left")
    return {
        "tmax_korelasyon": m["TEMP MÁX (ºC)"].corr(m["tmax"]),
        "yagis_korelasyon": m["PRECIPITAÇÃO (mm)"].corr(m["rain"]),
        "yagis_spearman": m["PRECIPITAÇÃO (mm)"].corr(m["rain"],
                                                     method="spearman"),
    }
