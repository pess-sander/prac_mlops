class DataStreamer:
    def __init__(self, df, batch_size, timestamp_col):
        self.df = df.copy()
        self.batch_size = batch_size
        self.current_idx = 0

        self.df = self.df.sort_values(timestamp_col)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.df):
            raise StopIteration
        
        batch = self.df.iloc[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return batch