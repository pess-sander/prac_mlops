import pandas as pd


class DataSource:
    def __init__(self, config, logger, storage):
        self.sources = config['data']['sources']
        self.test_source = config['data']['test']
        self.logger = logger
        self.storage = storage

    def _get_processed_sources(self):
        try:
            df = pd.read_sql(
                "SELECT DISTINCT source_path FROM batches WHERE status = 'success'",
                self.storage.engine
            )
            return set(df["source_path"])
        except Exception:
            return set()

    def load(self):
        processed = self._get_processed_sources()

        data = []

        for source in self.sources:
            path = source["path"]

            if path in processed:
                self.logger.info(f"Skipping {path} (already processed)")
                continue

            if source["type"] == "csv":
                try:
                    df = pd.read_csv(path)
                    data.append((path, df))
                except Exception as e:
                    self.logger.error(f"Failed to load {path}: {e}")
            else:
                self.logger.error(f"Unsupported source type: {source['type']}")

        return data
    
    def load_inference(self, in_path):
        try:
            df = pd.read_csv(in_path)
            return df
        except Exception as e:
            self.logger.error(f'Failed to load test set {in_path}: {e}')