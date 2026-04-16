from typing import Optional, Sequence

from src.model.streaming_model import StreamingModel


class Inference:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model = None
    
    def run_model(self, weights, X_num, X_artists, X_names):

        if not weights:
            raise ValueError("Model path is empty")

        self.model = StreamingModel(
            self.logger,
            None,
            model_path=weights
        )

        if not self.model.models:
            raise ValueError(f"No models loaded from {weights}")

        return self.model.predict(X_num, X_artists, X_names)
