class StatsCalculator:
    def compute_batch_meta(self, batch, batch_id):
        num_numeric = len(batch.select_dtypes(include="number").columns)
        num_categorical = len(batch.select_dtypes(exclude="number").columns)

        return {
            "batch_id": batch_id,
            "num_rows": int(len(batch)),
            "num_cols": int(batch.shape[1]),
            "num_missing": int(batch.isna().sum().sum()),
            "num_numeric": int(num_numeric),
            "num_categorical": int(num_categorical),
        }