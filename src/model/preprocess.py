from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col: str = "datum"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
        X = X.sort_values(self.date_col)

        X["hour"] = X[self.date_col].dt.hour
        X["day"] = X[self.date_col].dt.day
        X["month"] = X[self.date_col].dt.month
        X["dayofweek"] = X[self.date_col].dt.dayofweek

        X = X.drop(columns=[self.date_col])

        return X


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size: int = 24,
        target_col: str = "vodostaj",
        feature_cols: Optional[list[str]] = None,
    ):
        self.window_size = window_size
        self.target_col = target_col
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.feature_cols is not None:
            columns = self.feature_cols
        else:
            columns = list(X.columns)

        data = X[columns].to_numpy(dtype=float)

        target_idx = columns.index(self.target_col)

        X_windows = []
        y_values = []

        for i in range(self.window_size, len(data)):
            X_windows.append(data[i - self.window_size:i])
            y_values.append(data[i, target_idx])

        return np.array(X_windows), np.array(y_values)
