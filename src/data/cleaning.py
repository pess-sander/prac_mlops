import pandas as pd

class DataCleaner:
    def __init__(self, config):
        self.max_row_missing = config['cleaning']['max_row_missing']

    def clean(self, batch):
        df = batch.copy()

        df = df[df.isna().mean(axis=1) < self.max_row_missing]

        return df