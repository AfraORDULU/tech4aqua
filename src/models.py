"""
Tech4Aqua - Model tanimlari.

Ayarlar ara rapordaki deneylerle aynidir. Kucuk veride asiri
iyimserligi onlemek icin hiperparametre aramasi yapilmaz.
Eksik deger doldurma ve olcekleme Pipeline icinde oldugu icin
yalnizca egitim verisinden ogrenilir.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("LR", "RF", "LGBM")
DEEP_MODEL_NAMES = ("MLP", "GROUPDRO", "CORAL", "CNN1D")
SEQUENCE_MODEL_NAMES = ("GRU_SEQ", "CNN_SEQ", "GRU_HYB",
                        "GRU_SEL", "CNN_SEL", "GRU_HYB_SEL")


def make_model(name, seed=42):
    if name == "LR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                solver="liblinear", random_state=seed)),
        ])
    if name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=seed, n_jobs=-1)),
        ])
    if name == "LGBM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                objective="binary", n_estimators=200, learning_rate=0.05,
                num_leaves=7, max_depth=4, min_child_samples=15,
                reg_lambda=1.0, class_weight="balanced",
                random_state=seed, verbose=-1)),
        ])
    if name in DEEP_MODEL_NAMES:
        from src.deep import TorchTabularClassifier  # torch yalnizca gerekirse
        method = {"MLP": "erm", "GROUPDRO": "groupdro",
                  "CORAL": "coral", "CNN1D": "cnn1d"}[name]
        return TorchTabularClassifier(method=method, seed=seed)
    if name in SEQUENCE_MODEL_NAMES:
        from src import sequence  # torch yalnizca gerekirse
        store = (sequence.STORE_HYB if name.startswith("GRU_HYB")
                 else sequence.STORE)
        if store is None:
            raise RuntimeError("Once sequence.STORE / STORE_HYB ayarlayin")
        kind = "cnn" if name.startswith("CNN") else "gru"
        if name.endswith("_SEL"):
            return sequence.SelectedSequenceClassifier(kind=kind, store=store,
                                                       seed=seed)
        return sequence.SequenceClassifier(kind=kind, store=store, seed=seed)
    if name == "TABPFN":
        import os
        from tabpfn import TabPFNClassifier
        path = os.environ.get("TABPFN_MODEL_PATH", "auto")
        return TabPFNClassifier(model_path=path, device="cpu",
                                random_state=seed)
    raise ValueError(f"Bilinmeyen model: {name}")
