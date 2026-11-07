"""
Tech4Aqua - Dogrulama (train/test bolme) protokolleri.

1) grouped_splits: ZAPE gruplu, katmanli K-katli capraz dogrulama.
   Ayni ZAPE hem egitimde hem testte bulunamaz (gorulmemis bolge).
   purge=True ise test numunesiyle ayni belediye-gunu paylasan
   numuneler o katta egitimden cikarilir (arindirilmis CV). Ayni gun
   alinan numunelerin hava verisi neredeyse ayni oldugu icin bu,
   hava uzerinden sizintiyi engeller.

2) temporal_splits: Gecmis yillarla egit, sonraki yili test et
   (genisleyen pencere). unseen_zape=True ise test yilindaki ZAPE'ler
   egitimden tamamen cikarilir: hem gelecek hem gorulmemis bolge.

Her bolme, kurallarin saglandigini kodla dogrular (assert).
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _day_keys(df, idx):
    return set(zip(df["municipality"].values[idx], df["date"].values[idx]))


def purge_same_day(df, train_idx, test_idx):
    """Test ile ayni belediye-gunu paylasan egitim satirlarini cikarir."""
    test_keys = _day_keys(df, test_idx)
    keys = list(zip(df["municipality"].values[train_idx],
                    df["date"].values[train_idx]))
    keep = np.array([k not in test_keys for k in keys], dtype=bool)
    return train_idx[keep]


def _check(df, train_idx, test_idx, purge):
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    tr_z = set(df["zape"].values[train_idx])
    te_z = set(df["zape"].values[test_idx])
    assert not (tr_z & te_z), "Egitim ve testte ortak ZAPE var"
    if purge:
        assert not (_day_keys(df, train_idx) & _day_keys(df, test_idx)), \
            "Egitim ve testte ortak belediye-gun var"


def grouped_splits(df, n_splits=5, seed=0, purge=True, index=None):
    """(train_idx, test_idx) ciftleri uretir. index verilirse alt kume."""
    index = np.arange(len(df)) if index is None else np.asarray(index)
    sub = df.iloc[index]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    for tr, te in cv.split(np.zeros(len(sub)), sub["y"], sub["zape"]):
        tr, te = index[tr], index[te]
        if purge:
            tr = purge_same_day(df, tr, te)
        _check(df, tr, te, purge)
        yield tr, te


def temporal_splits(df, test_years, unseen_zape=False, min_train_pos=5,
                    min_train_zapes=5):
    """Her test yili icin (yil, train_idx, test_idx) uretir.

    Egitim: test yilindan ONCEKI tum yillar. Egitimde en az
    min_train_pos pozitif / min_train_zapes ZAPE yoksa veya testte hic
    pozitif yoksa o yil atlanir.
    """
    years = df["year"].values
    for year in test_years:
        te = np.where(years == year)[0]
        tr_mask = years < year
        if unseen_zape:
            tr_mask &= ~df["zape"].isin(set(df["zape"].values[te])).values
        tr = np.where(tr_mask)[0]
        if df["y"].values[tr].sum() < min_train_pos:
            continue
        if len(set(df["zape"].values[tr])) < min_train_zapes:
            continue
        if df["y"].values[te].sum() == 0:
            continue
        assert df["date"].values[tr].max() < df["date"].values[te].min()
        if unseen_zape:
            _check(df, tr, te, purge=False)
        yield year, tr, te
