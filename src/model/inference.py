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
        
        preds = []
		
        for i in range(len(X_num)):
            y_pred = self.model.predict(
                X_num[i:i+1],
                X_artists[i:i+1] if X_artists is not None else None,
                X_names[i:i+1] if X_names is not None else None
            )
			
            preds.append(y_pred)

        return preds
