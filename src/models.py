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
    raise ValueError(f"Bilinmeyen model: {name}")
