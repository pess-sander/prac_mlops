from src.data.source import DataSource
from src.data.streamer import DataStreamer
from src.data.storage import DataStorage
from src.data.stats import StatsCalculator
from src.data.cleaning import DataCleaner
from src.data.eda import EDAReporter
from src.data.feature_engineering import FeatureEngineer
from src.data.prepare_data import PrepareData

from src.model.train import Trainer

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

    def run(self, mode, weights):
        if mode == 'update':
            self.run_update()
        elif mode == 'inference':
            self.run_inference(weights)
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

        for source_path, df in sources_data:
            streamer = DataStreamer(df, self.batch_size, self.timestamp_col)

            for batch_id, batch in enumerate(streamer):
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
                    
                    # train and validate
                    self.trainer.train_and_validate(X_num, X_artists, X_names, y, batch_id)

                except Exception as e:
                    self.logger.error(f"Unexpected failure on batch {batch_id}: {e}")

        self.logger.info("Update pipeline finished")


    def run_inference(self, weights):
        pass

    def run_summary(self):
        pass