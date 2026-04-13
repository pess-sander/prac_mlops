from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd


class DataStorage:
    def __init__(self, config):
        db_conf = config["storage"]["postgres"]

        self.engine = create_engine(
            f"postgresql://{db_conf['user']}:{db_conf['password']}"
            f"@{db_conf['host']}:{db_conf['port']}/{db_conf['database']}"
        )

    def save_raw(self, batch, batch_id, source_path):
        batch = batch.copy()

        ingestion_time = datetime.utcnow()

        meta = {
            "batch_id": batch_id,
            "source_path": source_path,
            "ingestion_time": ingestion_time,
            "rows_count": len(batch),
            "status": None,
            "message": None
        }

        try:
            batch["batch_id"] = batch_id
            batch["ingestion_time"] = ingestion_time

            batch = batch.where(pd.notnull(batch), None)

            # 0.0/1.0 -> False/True
            if "explicit" in batch.columns:
                batch["explicit"] = batch["explicit"].apply(
                    lambda x: None if pd.isna(x) else bool(int(x))
                )

            if "row_id" in batch.columns:
                batch = batch.drop(columns=["row_id"])

            batch.to_sql(
                "raw_data",
                self.engine,
                if_exists="append",
                index=False
            )

            meta["status"] = "success"
            meta["message"] = "batch inserted successfully"

        except Exception as e:
            meta["status"] = "failed"
            meta["message"] = f"{type(e).__name__}: {str(e)[:300]}"

            print(f"[ERROR] Batch {batch_id} failed: {e}")

        finally:
            df = pd.DataFrame([meta])
            df.to_sql("batches", self.engine, if_exists="append", index=False)

    def save_data_quality(self, dq_list):
        df = pd.DataFrame(dq_list)
        df.to_sql("data_quality", self.engine, if_exists="append", index=False)