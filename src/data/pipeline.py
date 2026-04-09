from src.data.source import DataSource
from src.data.streamer import DataStreamer
from src.data.storage import DataStorage
from src.data.stats import StatsCalculator

class DataCollectionPipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.batch_size = self.config['collection']['batch_size']
        self.target_col = self.config['data']['target_column']
        self.timestamp_col = self.config['data']['timestamp_column']

        self.source = DataSource(config, logger)
        self.storage = DataStorage(config)
        self.stats = StatsCalculator()

    def run(self, mode, weights):
        if mode == 'update':
            self.run_update()
        elif mode == 'inference':
            self.run_inference(weights)
        elif mode == 'summary':
            self.run_summary()
        self.logger.error(f'Unsupported mode: {mode}')
        raise ValueError(f'Unsupported mode: {mode}')
    
    def run_update(self):
        self.logger.info('Update pipeline started')

        df = self.source.load()
        if df is None or df.empty():
            self.logger.info('No new data')
            return
        
        streamer = DataStreamer(df, self.batch_size, self.timestamp_col)
        for idx, batch in enumerate(streamer):
            self.logger.info(f'Processing batch {idx}: {len(batch)} rows')

            try:
                self.storage.save_raw(batch, idx)

            except Exception as e:
                self.logger.error(f'Failed on batch {idx}: \n\t{e}')


    def run_inference(self, weights):
        pass

    def run_summary(self):
        pass