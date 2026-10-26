import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

def oof_predictions(model, X, y, groups, seed, n_splits=5):
    probabilities = np.full(len(y), np.nan)

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for train_idx, test_idx in cv.split(X, y, groups):
        fitted_model = clone(model)
        fitted_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        probabilities[test_idx] = fitted_model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

    if np.isnan(probabilities).any():
        raise ValueError("Bazı örnekler için tahmin oluşturulamadı.")

    return probabilities


def summarize(y, probabilities, threshold=0.5):
    predictions = (probabilities > threshold).astype(int)

    return {
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(y, probabilities),
        "MCC": matthews_corrcoef(y, predictions),
    }