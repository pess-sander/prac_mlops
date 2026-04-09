import pandas as pd
import json
import os

class DataSource:
    def __init__(self, config, logger):
        self.sources = config['data']['sources']
        self.processed_file = config['data']['processed_data']

        self.logger = logger

    def load(self):
        dfs = []

        if os.path.exists(self.processed_file):
            with open(self.processed_file) as f:
                processed = set(json.load(f))
        else:
            processed = set()

        for source in self.sources:
            if source['path'] in processed:
                self.logger.info(f'Skipping {source["path"]} because it was previously processed')
                continue

            if source['type'] == 'csv':
                try:
                    df = pd.read_csv(source['path'])
                except Exception as e:
                    self.logger.error(f'Failed to load {source["path"]}: \n\t{e}')
                
                dfs.append(df)
                processed.append(source['path'])
            else:
                self.logger.error(f'Unsupported source type: {source['type']}')
        
        with open(self.processed_file, 'w') as f:
            json.dump(processed, f, indent=4)

        return pd.concat(dfs, ignore_index=True)