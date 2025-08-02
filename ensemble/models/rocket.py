import os
import pickle


class Rocket:
    def __init__(self, model_dir: str, n_models: int):
        self.n_models = n_models
        self.models = self._load_model(model_dir)

    def _load_model(self, model_dir):
        models = []
        for i in range(1, self.n_models + 1):
            with open(os.path.join(model_dir, f"rocket_tsc_{i}.pkl"), "rb") as f:
                model = pickle.load(f)
                models.append(model)
        return models

    def predict(self, x, t):
        if t > self.n_models:
            raise ValueError("Time step exceeds number of models")
        return self.models[t - 1].predict(x)

    def predict_proba(self, x, t):
        if t > self.n_models:
            raise ValueError("Time step exceeds number of models")
        return self.models[t - 1].predict_proba(x)