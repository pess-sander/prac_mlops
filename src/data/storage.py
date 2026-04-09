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

    def save_raw(self, batch, batch_id):
        batch = batch.copy()
        batch["batch_id"] = batch_id
        batch["ingestion_time"] = datetime.utcnow()

        batch.to_sql("raw_data", self.engine, if_exists="append", index=False)

    def save_batch_meta(self, meta):
        df = pd.DataFrame([meta])
        df.to_sql("batches", self.engine, if_exists="append", index=False)

    def save_data_quality(self, dq_list):
        df = pd.DataFrame(dq_list)
        df.to_sql("data_quality", self.engine, if_exists="append", index=False)