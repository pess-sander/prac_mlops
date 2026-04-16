from collections import deque
from river import linear_model, optim
import pickle
import os




class StreamingModel:
	def __init__(self, logger, config,
			 model_path=None,
			 window_size=2000,
			 prune_every=500,
			 min_models=3):

		self.logger = logger
		self.window_size = window_size
		self.prune_every = prune_every
		self.min_models = min_models
		self.model_path = model_path

		self.models = []
		# self.weights = []
		self.errors = []
		self.features_config = []

		self.step = 1

		# init models
		if config is not None:
			for cfg in config:
				reg_type = cfg.get("reg_type", "l2")
				reg = cfg.get("reg", 0.01)
				lr = cfg.get("lr", 0.01)

				if reg_type == "l1":
					model = linear_model.LinearRegression(
						optimizer=optim.SGD(lr=lr),
						l1=reg,
						l2=0.0,
						intercept_lr=lr
					)
				elif reg_type == "l2":
					model = linear_model.LinearRegression(
						optimizer=optim.SGD(lr=lr),
						l1=0.0,
						l2=reg,
						intercept_lr=lr
					)
				else:
					raise ValueError(f"Unsupported reg_type: {reg_type}")
			   
				self.models.append(model)
				# self.weights.append(1.0)
				self.errors.append(deque(maxlen=window_size))
				self.features_config.append(cfg.get('features', ['num', 'artists', 'names']))

		if os.path.exists(model_path):
			self.load_models(model_path)


	def _to_river_dict(self, X_num, X_artists, X_text, features):
		x = {}
		
		if 'num' in features:
			for i, v in enumerate(X_num[0]):
				x[f"num_{i}"] = float(v)
		
		if 'artists' in features and X_artists is not None:
			for i, v in enumerate(X_artists[0]):
				x[f"artist_{i}"] = float(v)
		
		if 'names' in features and X_text is not None:
			for i, v in enumerate(X_text[0]):
				x[f"text_{i}"] = float(v)
		
		return x

	def _prune_models(self):


		if len(self.models) <= self.min_models:
			return


		# compute scores
		scores = []
		for i in range(len(self.models)):
			if len(self.errors[i]) == 0:
				scores.append(float("inf"))
			else:
				mae = sum(self.errors[i]) / len(self.errors[i])
				scores.append(mae)


		# worst model index
		worst_idx = max(range(len(scores)), key=lambda i: scores[i])


		# safety check
		if len(self.models) > self.min_models:
			removed = self.models[worst_idx]
			self.logger.info(
				f"[PRUNE] Removing model idx={worst_idx}, MAE={scores[worst_idx]:.4f}"
			)


			del self.models[worst_idx]
			# del self.weights[worst_idx]
			del self.errors[worst_idx]
			del self.features_config[worst_idx]


	def predict(self, X_num, X_artists, X_text):
		preds = []

		for i, m in enumerate(self.models):
			features = self.features_config[i]
			x = self._to_river_dict(X_num, X_artists, X_text, features)
			preds.append(m.predict_one(x))

		if not preds:
			return 0.0

		return sum(preds) / len(preds)

	def learn(self, X_num, X_artists, X_text, y):
		self.step += 1
		y_true = float(y[0])

		preds = []
		xs = []

		for i, m in enumerate(self.models):
			features = self.features_config[i]
			x = self._to_river_dict(X_num, X_artists, X_text, features)
			xs.append(x)
			preds.append(m.predict_one(x))

		for i, m in enumerate(self.models):
			err = abs(y_true - preds[i])
			self.errors[i].append(err)
			m.learn_one(xs[i], y_true)

		if (self.step % self.prune_every == 0) and (self.step > 5000):
			self._prune_models()

	def validate_stream(self, X_num, X_artists, X_text, y):
		maes = []
		
		for i in range(len(y)):
			y_true = float(y[i])
			y_pred = self.predict(
				X_num[i:i+1],
				X_artists[i:i+1] if X_artists is not None else None,
				X_text[i:i+1] if X_text is not None else None
			)
			
			maes.append(abs(y_true - y_pred))
			
			for j, m in enumerate(self.models):
				features = self.features_config[j]
				x = self._to_river_dict(
					X_num[i:i+1],
					X_artists[i:i+1] if X_artists is not None else None,
					X_text[i:i+1] if X_text is not None else None,
					features
				)
				m.learn_one(x, y_true)
		
		mae = sum(maes) / len(maes)
		self.logger.info(f"[ENSEMBLE+PRUNE VALIDATION] MAE = {mae:.4f}")
		
		return mae
	
	def save_models(self, filepath):

		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		self.logger.info(f"Ensemble has {len(self.models)} models:")
		for i, (model, feat_cfg) in enumerate(zip(self.models, self.features_config)):
			optimizer = getattr(model, "optimizer", None)
			if isinstance(optimizer, optim.FTRLProximal):
				self.logger.info(
					f"Model {i:2d} | "
					f"alpha={optimizer.alpha:.4f}, "
					f"beta={optimizer.beta:.4f}, "
					f"l1={optimizer.l1:.4f}, "
					f"l2={optimizer.l2:.4f} | "
					f"features={feat_cfg}"
				)
			else:
				self.logger.info(
					f"Model {i:2d} | "
					f"optimizer_type={type(optimizer).__name__} | "
					f"optimizer_state={getattr(optimizer, '__dict__', {})} | "
					f"l1={getattr(model, 'l1', None)} | "
					f"l2={getattr(model, 'l2', None)} | "
					f"intercept_lr={getattr(model, 'intercept_lr', None)} | "
					f"features={feat_cfg}"
				)

		models_data = {
			'models': self.models,
			# 'weights': self.weights,
			'errors': [list(err) for err in self.errors],
			'features_config': self.features_config,
			'step': self.step
		}

		with open(filepath, 'wb') as f:
			pickle.dump(models_data, f)

		self.logger.info(f"Saved {len(self.models)} models to {filepath}")

	def load_models(self, filepath):
		if not filepath or not os.path.exists(filepath):
			self.logger.info(f"[LOAD] No model file found at {filepath}. Starting fresh.")
			return False


		with open(filepath, 'rb') as f:
			data = pickle.load(f)


		self.models = data['models']
		# self.weights = data['weights']
		self.errors = [deque(err, maxlen=self.window_size) for err in data['errors']]
		self.features_config = data['features_config']
		self.step = data.get('step', 1)


		self.logger.info(
			f"[LOAD] Restored {len(self.models)} models from {filepath}, step={self.step}"
		)


		return True


