import pandas as pd
import numpy as np
from datetime import datetime


class StatsCalculator:
    def __init__(self):
        self.categorical_cols = {"year", "key", "mode"}

    def compute_data_quality(self, batch: pd.DataFrame, batch_id: int):
        results = []
        n = len(batch)
        now = datetime.utcnow()

        for col in batch.columns:
            series = batch[col]

            missing_ratio = series.isna().mean()
            unique_ratio = series.nunique(dropna=True) / n if n > 0 else None

            results.append(self._row(batch_id, col, "missing_ratio", missing_ratio, now))
            results.append(self._row(batch_id, col, "unique_ratio", unique_ratio, now))

            is_categorical = (
                col in self.categorical_cols
                or series.dtype == "object"
            )

            if is_categorical:
                vc = series.value_counts(dropna=True)

                if len(vc) > 0:
                    top_value = vc.index[0]
                    top_ratio = vc.iloc[0] / n
                else:
                    top_value = None
                    top_ratio = None

                results.append(self._row(batch_id, col, "top_value_ratio", top_ratio, now))

                results.append({
                    "batch_id": batch_id,
                    "column_name": col,
                    "metric_name": "top_value",
                    "metric_value": None,
                    "metric_text": str(top_value) if top_value is not None else None,
                    "created_at": now
                })

            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()

                if len(clean) > 0:
                    results.append(self._row(batch_id, col, "mean", clean.mean(), now))
                    results.append(self._row(batch_id, col, "std", clean.std(), now))
                    results.append(self._row(batch_id, col, "min", clean.min(), now))
                    results.append(self._row(batch_id, col, "max", clean.max(), now))

        return results

    def _row(self, batch_id, col, metric, value, ts):
        return {
            "batch_id": batch_id,
            "column_name": col,
            "metric_name": metric,
            "metric_value": None if pd.isna(value) else float(value),
            "metric_text": None,
            "created_at": ts
        }