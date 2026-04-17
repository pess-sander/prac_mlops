from src.data.source import DataSource
from src.data.streamer import DataStreamer
from src.data.storage import DataStorage
from src.data.stats import StatsCalculator
from src.data.cleaning import DataCleaner
from src.data.eda import EDAReporter
from src.data.feature_engineering import FeatureEngineer
from src.data.prepare_data import PrepareData

from src.model.train import Trainer
from src.model.inference import Inference

import pandas as pd
import os

class DataCollectionPipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.batch_size = self.config['collection']['batch_size']
        self.target_col = self.config['data']['target_column']
        self.timestamp_col = self.config['data']['timestamp_column']

        self.storage = DataStorage(config)
        self.source = DataSource(config, logger, self.storage)
        self.stats = StatsCalculator()
        self.cleaner = DataCleaner(config)
        self.eda = EDAReporter(config)
        self.fe = FeatureEngineer()
        self.prepare = PrepareData()

        self.trainer = Trainer(config, logger)
        self.inference = Inference(config, logger)

    def run(self, mode, weights, in_file):
        if mode == 'update':
            self.run_update()
        elif mode == 'inference':
            self.run_inference(weights, in_file)
        elif mode == 'summary':
            self.run_summary()
        else:
            self.logger.error(f'Unsupported mode: {mode}')
            raise ValueError(f'Unsupported mode: {mode}')
    
    def run_update(self):
        self.logger.info("Update pipeline started")

        sources_data = self.source.load()

        if not sources_data:
            self.logger.info("No new data")
            return False

        query = "SELECT MAX(batch_id) AS max_id FROM batches"
        result = pd.read_sql(query, self.storage.engine)
        last_batch_id = result["max_id"].iloc[0]
        batch_id = int(last_batch_id) + 1 if pd.notnull(last_batch_id) else 0
        for source_path, df in sources_data:
            streamer = DataStreamer(df, self.batch_size, self.timestamp_col)

            for batch in streamer:
                self.logger.info(f"Processing batch {batch_id} from {source_path}")

                try:
                    self.storage.save_raw(batch, batch_id, source_path)

                    dq_list = self.stats.compute_data_quality(batch, batch_id)
                    self.storage.save_data_quality(dq_list)

                    clean_batch = self.cleaner.clean(batch)

                    # feature engineering
                    clean_batch = self.fe.transform(clean_batch)

                    # EDA plots
                    paths = self.eda.generate(clean_batch, batch_id)

                    self.logger.info(f"Saved {len(paths)} plots")

                    # Prepare Data
                    X_num, X_artists, X_names, y = self.prepare.preprocess(clean_batch)

                    self.logger.info(f"X_num {X_num.max()}")
                    self.logger.info(f"X_artists {X_artists.max()}")
                    self.logger.info(f"X_names {X_names.max()}")
                    
                    # train and validate
                    self.trainer.train_and_validate(X_num, X_artists, X_names, y, batch_id)

                except Exception as e:
                    self.logger.error(f"Unexpected failure on batch {batch_id}: {e}")

                finally:
                    batch_id += 1

        self.logger.info("Update pipeline finished")


    def run_inference(self, weights, in_path):
        self.logger.info("Inference pipeline started")

        in_data = self.source.load_inference(in_path)
        os.makedirs('results', exist_ok=True)
        out_path = os.path.join('results', os.path.basename(in_path))

        streamer = DataStreamer(in_data, self.batch_size, self.timestamp_col)
        all_rows = []

        for batch_id, batch in enumerate(streamer):
            self.logger.info(f"Processing batch {batch_id} from {in_path}")

            try:
                clean_batch = self.cleaner.clean(batch)

                # feature engineering
                clean_batch = self.fe.transform(clean_batch)

                # Prepare Data
                X_num, X_artists, X_names, y = self.prepare.preprocess(clean_batch)
                
                # run model
                predict = self.inference.run_model(weights, X_num, X_artists, X_names)
                batch_df = pd.DataFrame({
                    "name": clean_batch["name"].values,
                    "artists": clean_batch["artists"].values,
                    "predicted_popularity": predict
                })
                all_rows.append(batch_df)

            except Exception as e:
                self.logger.error(f"Unexpected failure on batch {batch_id}: {e}")
        
        if all_rows:
            final_df = pd.concat(all_rows, ignore_index=True)
            final_df.to_csv(out_path, index=False)
            self.logger.info(f"Saved {len(final_df)} predictions to {out_path}")
        else:
            self.logger.warning("No predictions to save")

        self.logger.info("Inference pipeline finished")

    def run_summary(self):
        self.logger.info("Summary pipeline started")

        import json
        os.makedirs("reports", exist_ok=True)

        summary = {}

        try:
            dq = pd.read_sql("SELECT * FROM data_quality", self.storage.engine)

            if not dq.empty:
                dq = dq[dq["metric_name"].isin(["missing_ratio", "unique_ratio"])]

                pivot = dq.pivot_table(
                    index=["batch_id", "column_name"],
                    columns="metric_name",
                    values="metric_value"
                ).reset_index()

                summary["data_quality"] = {
                    "avg_missing_ratio": float(pivot["missing_ratio"].mean()),
                    "avg_unique_ratio": float(pivot["unique_ratio"].mean()),
                    "max_missing_ratio": float(pivot["missing_ratio"].max()),
                    "min_unique_ratio": float(pivot["unique_ratio"].min()),
                }

                worst_missing = pivot.loc[pivot["missing_ratio"].idxmax()]
                worst_unique = pivot.loc[pivot["unique_ratio"].idxmin()]

                summary["worst_columns"] = {
                    "highest_missing": {
                        "column": worst_missing["column_name"],
                        "value": float(worst_missing["missing_ratio"]),
                        "batch_id": int(worst_missing["batch_id"]),
                    },
                    "lowest_unique": {
                        "column": worst_unique["column_name"],
                        "value": float(worst_unique["unique_ratio"]),
                        "batch_id": int(worst_unique["batch_id"]),
                    }
                }

            metrics_path = "data/metrics.csv"

            if os.path.exists(metrics_path):
                metrics = pd.read_csv(metrics_path)

                best_row = metrics.loc[metrics["mae"].idxmin()]

                summary["best_model"] = {
                    "mae": float(best_row["mae"]),
                    "model_idx": int(best_row["model_idx"]),
                    "optimizer": best_row.get("optimizer"),
                    "lr": best_row.get("lr"),
                    "l1": best_row.get("l1"),
                    "l2": best_row.get("l2"),
                    "intercept_lr": best_row.get("intercept_lr"),
                    "features": best_row.get("features"),
                    "timestamp": best_row.get("timestamp")
                }

                summary["metrics_overview"] = {
                    "num_records": len(metrics),
                    "avg_mae": float(metrics["mae"].mean()),
                    "min_mae": float(metrics["mae"].min()),
                    "max_mae": float(metrics["mae"].max()),
                }

            else:
                self.logger.warning("No metrics.csv found")

            batches = pd.read_sql("SELECT * FROM batches", self.storage.engine)

            if not batches.empty:
                summary["batches"] = {
                    "total": len(batches),
                    "success": int((batches["status"] == "success").sum()),
                    "failed": int((batches["status"] == "failed").sum()),
                }

            out_path = "reports/summary.json"

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(f"Summary saved to {out_path}")
            self.logger.info("Summary pipeline finished")

            return out_path

        except Exception as e:
            self.logger.error(f"Summary failed: {e}")
            return None