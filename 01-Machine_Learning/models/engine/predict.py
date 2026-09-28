"""Load final models and predict from ordered reactant descriptors.

Input columns: 1,875 amine descriptors followed by 1,875 aldehyde descriptors,
in the original descriptor order. Non-finite values use saved imputation values.
Output columns are probabilities for classes 0 and 1; class 1 is MIC < 0.5 mM.
Classify with probabilities[:, 1] >= the model's threshold in registry.csv.

Example, with this module on the Python import path:
    model = load_model(Path("models/XGBOOST/model.joblib"))
    probabilities = predict_probabilities(descriptors, model)
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, TypedDict

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Estimator: TypeAlias = (
    CatBoostClassifier | LGBMClassifier | XGBClassifier | ExtraTreesClassifier
    | GradientBoostingClassifier | HistGradientBoostingClassifier
    | RandomForestClassifier | LogisticRegression | GaussianNB
    | KNeighborsClassifier | MLPClassifier | SVC | DecisionTreeClassifier
    | LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis
)


class ModelState(TypedDict):
    """Fitted values required for prediction."""

    input_width: int
    columns: IntArray
    medians: FloatArray
    means: FloatArray
    scales: FloatArray
    estimator: Estimator
    calibrator: LogisticRegression | None


def load_model(path: Path) -> ModelState:
    """Load a trusted model file and validate its prediction state."""
    payload = joblib.load(path)
    required = {"input_width", "columns", "medians", "means", "scales", "estimator", "calibrator"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Model file lacks required prediction fields")
    width = payload["input_width"]
    columns = payload["columns"]
    if type(width) is not int or width <= 0:
        raise ValueError("Model input_width must be a positive integer")
    if not isinstance(columns, np.ndarray) or columns.dtype != np.int64 or columns.ndim != 1:
        raise TypeError("Model columns must be a one-dimensional int64 array")
    if not len(columns) or len(np.unique(columns)) != len(columns) or np.any((columns < 0) | (columns >= width)):
        raise ValueError("Model columns must be unique indices within input_width")
    for name in ("medians", "means", "scales"):
        values = payload[name]
        if not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.shape != columns.shape:
            raise TypeError(f"Model {name} must be a float64 array matching columns")
        if not np.isfinite(values).all():
            raise ValueError(f"Model {name} contains non-finite values")
    if np.any(payload["scales"] <= 0):
        raise ValueError("Model scales must be positive")
    estimator = payload["estimator"]
    calibrator = payload["calibrator"]
    if not isinstance(estimator, Estimator):
        raise TypeError(f"Unsupported estimator type: {type(estimator).__name__}")
    feature_count = len(estimator.feature_names_) if isinstance(estimator, CatBoostClassifier) else estimator.n_features_in_
    if not np.array_equal(estimator.classes_, [0, 1]) or feature_count != len(columns):
        raise ValueError("Estimator classes or feature count differ from prediction state")
    if isinstance(estimator, SVC):
        if not isinstance(calibrator, LogisticRegression):
            raise TypeError("SVC requires a fitted LogisticRegression probability calibrator")
        if calibrator.n_features_in_ != 1 or not np.array_equal(calibrator.classes_, [0, 1]):
            raise ValueError("Probability calibrator must accept one score and output classes 0 and 1")
    elif calibrator is not None:
        raise ValueError("Only SVC uses a separate probability calibrator")
    return ModelState(input_width=width, columns=columns, medians=payload["medians"],
                      means=payload["means"], scales=payload["scales"],
                      estimator=estimator, calibrator=calibrator)


def predict_probabilities(features: FloatArray, model: ModelState) -> FloatArray:
    """Apply saved preprocessing and return binary probabilities without refitting."""
    if features.ndim != 2 or features.shape[1] != model["input_width"]:
        raise ValueError(f"Expected {model['input_width']} input columns; received shape {features.shape}")
    raw = features[:, model["columns"]]
    imputed = np.where(np.isfinite(raw), raw, model["medians"])
    transformed = np.ascontiguousarray((imputed - model["means"]) / model["scales"], dtype=np.float64)
    if not np.isfinite(transformed).all():
        raise ArithmeticError("Feature transformation produced non-finite values")
    estimator = model["estimator"]
    calibrator = model["calibrator"]
    if calibrator is not None:
        scores = np.asarray(estimator.decision_function(transformed), dtype=np.float64).reshape(-1, 1)
        probabilities = calibrator.predict_proba(scores)
    elif isinstance(estimator, CatBoostClassifier):
        probabilities = estimator.predict_proba(transformed, thread_count=1)
    else:
        probabilities = estimator.predict_proba(transformed)
    result = np.asarray(probabilities, dtype=np.float64)
    if result.shape != (len(features), 2) or not np.isfinite(result).all():
        raise ArithmeticError("Model returned invalid probability dimensions or non-finite values")
    if np.any((result < 0) | (result > 1)) or not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ArithmeticError("Model probabilities must be within [0, 1] and sum to one")
    return result
