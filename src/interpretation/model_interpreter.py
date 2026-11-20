from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(
    trained_model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str = "f1_weighted",
    repeats: int = 10
) -> pd.DataFrame:
    if len(features) != len(target):
        raise ValueError("Размеры признаков и целевой переменной не совпадают")
    result = permutation_importance(
        trained_model,
        features,
        target,
        scoring=scoring,
        n_repeats=repeats,
        random_state=42,
        n_jobs=-1
    )
    return pd.DataFrame({
        "feature": features.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)


def explain_prediction(
    trained_model: Any,
    data: pd.DataFrame,
    row_index: int
) -> dict:
    if row_index < 0 or row_index >= len(data):
        raise IndexError("Индекс объекта находится за пределами данных")
    selected_row = data.iloc[[row_index]]
    prediction = trained_model.predict(selected_row)[0]
    result: dict[str, Any] = {
        "row_index": int(row_index),
        "prediction": int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction)
    }
    if hasattr(trained_model, "predict_proba"):
        probabilities = trained_model.predict_proba(selected_row)[0]
        result["probabilities"] = [round(float(v), 4) for v in probabilities]
    return result


class ModelInterpreter:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")
        self.model = joblib.load(path)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def global_importance(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        scoring: str = "f1_weighted"
    ) -> pd.DataFrame:
        return calculate_permutation_importance(self.model, data, target, scoring)

    def explain_row(self, data: pd.DataFrame, row_index: int) -> dict:
        return explain_prediction(self.model, data, row_index)
