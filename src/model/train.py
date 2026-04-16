from typing import Optional, Sequence

from src.model.streaming_model import StreamingModel


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.model_config = config.get('model', {})
        self.model_grid = self.model_config.get('model_grid', {}).get('variants', [])
        self.feature_combinations = self.model_config.get('feature_combinations', [])
        self.streaming_params = self.model_config.get('streaming_model', {})
        self.train_config = config.get('training', {})

        self.model = None


    def _create_model_grid(self):
        extended_model_grid = []

        for feature_combo in self.feature_combinations:
            for cfg in self.model_grid:
                extended_model_grid.append({
                    **cfg,
                    'features': feature_combo
                })

        self.logger.info(f"Created {len(extended_model_grid)} model configurations")
        self.logger.info(f"Feature combinations: {self.feature_combinations}")
        self.logger.info(f"Model variants: {len(self.model_grid)}")

        return extended_model_grid


    def train_and_validate(
        self,
        X_num: Sequence,
        X_artists: Optional[Sequence],
        X_names: Optional[Sequence],
        y: Sequence,
        batch_id: int,
    ) -> float:

        # split
        val_split = self.train_config.get('validation_split', 0.8)
        split_idx = int(len(y) * val_split)

        X_num_tr, X_num_val = X_num[:split_idx], X_num[split_idx:]
        X_art_tr, X_art_val = (
            (X_artists[:split_idx], X_artists[split_idx:]) if X_artists is not None else (None, None)
        )
        X_text_tr, X_names_val = (
            (X_names[:split_idx], X_names[split_idx:]) if X_names is not None else (None, None)
        )
        y_tr, y_val = y[:split_idx], y[split_idx:]

        # Create / load model lazily
        if self.model is None:
            extended_model_grid = self._create_model_grid()

            model_path = self.model_config.get('streaming_model', {}).get(
                'save_path',
                'models/streaming_ensemble.pkl'
            )

            self.model = StreamingModel(
                self.logger,
                extended_model_grid,
                model_path=model_path,
                window_size=self.streaming_params.get('window_size', 2000),
                prune_every=self.streaming_params.get('prune_every', 2000),
                min_models=self.streaming_params.get('min_models', 3),
            )

            self.logger.info(f"Model created with {len(extended_model_grid)} base models")
            self.logger.info(
                f"Streaming params: window_size={self.streaming_params.get('window_size', 2000)}, "
                f"prune_every={self.streaming_params.get('prune_every', 2000)}, "
                f"min_models={self.streaming_params.get('min_models', 3)}"
            )

        # Train
        for i in range(len(y_tr)):
            self.model.learn(
                X_num_tr[i:i+1],
                X_art_tr[i:i+1] if X_art_tr is not None else None,
                X_text_tr[i:i+1] if X_text_tr is not None else None,
                y_tr[i:i+1]
            )

        # Validate
        val_mae = self.model.validate_stream(
            X_num_val,
            X_art_val,
            X_names_val,
            y_val
        )

        self.logger.info(f"Batch {batch_id} VAL MAE = {val_mae:.4f}")

        # Save ensemble
        model_path = self.model_config.get('streaming_model', {}).get(
            'save_path',
            'models/streaming_ensemble.pkl'
        )
        self.model.save_models(model_path)

        return val_mae