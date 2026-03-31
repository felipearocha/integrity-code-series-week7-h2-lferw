"""
INTEGRITY CODE SERIES -- Week 7
ML Surrogate: Gradient Boosted Regression for Rapid Screening

Purpose: Train a GBR surrogate on the LHS parametric sweep data
to enable rapid go/no-go screening of pipeline segments for H2
conversion without running the full physics solver.

The surrogate is a TOOL, not the analysis. It is trained on
physics-generated data and validated against held-out physics runs.
Feature importances serve as a sanity check: if the surrogate
ranks parameters differently than physical reasoning predicts,
that signals a problem with either the training data or the model.

Expected physical ranking (prior expectation):
    1. K_IC_seam (dominant: controls failure threshold)
    2. pit_depth (initial damage state)
    3. p_H2 (hydrogen concentration driver)
    4. D_L (transport kinetics)
    5. aspect_ratio (geometry modifier)
    6. f_seam (pit growth history)

If the surrogate disagrees significantly, investigate.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


@dataclass
class SurrogateResult:
    """Container for surrogate model results."""
    model: GradientBoostingRegressor
    r2_train: float
    r2_test: float
    mae_train: float
    mae_test: float
    feature_importances: Dict[str, float]
    param_names: list
    y_test: np.ndarray
    y_pred_test: np.ndarray
    X_test: np.ndarray


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    param_names: list,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> SurrogateResult:
    """
    Train GBR surrogate on physics-generated data.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input parameters from LHS sweep.
    y : ndarray, shape (n_samples,)
        Remaining life in years from physics solver.
    param_names : list of str
    test_fraction : float
    seed : int

    Returns
    -------
    SurrogateResult
    """
    # Filter out error cases
    valid = y > 0
    X_valid = X[valid]
    y_valid = y[valid]

    if len(y_valid) < 20:
        raise ValueError(f"Only {len(y_valid)} valid samples. Need at least 20.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=test_fraction, random_state=seed
    )

    # GBR with moderate complexity to avoid overfitting
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    feature_importances = {}
    for name, imp in zip(param_names, model.feature_importances_):
        feature_importances[name] = float(imp)

    return SurrogateResult(
        model=model,
        r2_train=float(r2_score(y_train, y_pred_train)),
        r2_test=float(r2_score(y_test, y_pred_test)),
        mae_train=float(mean_absolute_error(y_train, y_pred_train)),
        mae_test=float(mean_absolute_error(y_test, y_pred_test)),
        feature_importances=feature_importances,
        param_names=param_names,
        y_test=y_test,
        y_pred_test=y_pred_test,
        X_test=X_test,
    )


def validate_feature_ranking(importances: Dict[str, float]) -> Dict[str, str]:
    """
    Compare surrogate feature ranking against physical expectation.

    Expected top-3: K_IC_seam, pit_depth, p_H2 (in some order).
    If a physically secondary parameter dominates, flag it.

    Returns dict of parameter: status ('OK' or 'FLAG: ...')
    """
    sorted_features = sorted(importances.items(), key=lambda x: -x[1])
    top3_names = [f[0] for f in sorted_features[:3]]

    expected_top = {"K_IC_seam", "pit_depth_m", "p_H2_MPa"}
    result = {}

    for name, imp in importances.items():
        if name in expected_top:
            if name in top3_names:
                result[name] = "OK"
            else:
                result[name] = f"FLAG: expected top-3 but ranked {[f[0] for f in sorted_features].index(name)+1}"
        else:
            if name in top3_names:
                result[name] = f"FLAG: unexpected top-3 ranking (importance={imp:.3f})"
            else:
                result[name] = "OK"

    return result


def save_surrogate(surrogate: SurrogateResult, filepath: str):
    """Save trained surrogate to pickle."""
    with open(filepath, "wb") as f:
        pickle.dump({
            "model": surrogate.model,
            "param_names": surrogate.param_names,
            "r2_test": surrogate.r2_test,
            "feature_importances": surrogate.feature_importances,
        }, f)


def load_surrogate(filepath: str) -> dict:
    """Load trained surrogate from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
