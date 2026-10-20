"""Saving/loading of the prediction artifact: a logistic regression plus everything needed to use it."""
import json
from typing import List

import numpy as np

from features import COMPETITIONS, FEATURE_COLUMNS, LABELS, match_features

MODEL_PATH = "model.json"


def save_artifact(path, model, scaler, team_states, data_as_of, metrics):
    """Save the coefficients with the scaler, feature order and team states, as plain JSON."""
    with open(path, "w") as f:
        json.dump({
            "model_type": "logistic_regression",
            "C": float(model.C),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "classes": [int(c) for c in model.classes_],
            "feature_columns": list(FEATURE_COLUMNS),
            "scaler_mean": [float(v) for v in scaler.mean_],
            "scaler_scale": [float(v) for v in scaler.scale_],
            "team_states": team_states,
            "data_as_of": data_as_of,
            "metrics": metrics,
        }, f)


class Predictor:
    def __init__(self, path=MODEL_PATH):
        with open(path) as f:
            artifact = json.load(f)
        if artifact["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError(
                f"{path} was trained on different features. Run `python train.py` to retrain."
            )

        self.feature_columns = artifact["feature_columns"]
        self.C = artifact["C"]
        self.team_states = artifact["team_states"]
        self.data_as_of = artifact["data_as_of"]
        self.metrics = artifact["metrics"]
        self.coef = np.array(artifact["coef"])
        self.intercept = np.array(artifact["intercept"])
        self.classes = artifact["classes"]
        self.mean = np.array(artifact["scaler_mean"])
        self.scale = np.array(artifact["scaler_scale"])

    @property
    def input_size(self):
        return len(self.feature_columns)

    def teams_by_competition(self):
        """{competition code: sorted team names} for each competition's latest season."""
        latest = {}
        for state in self.team_states.values():
            code = state["competition"]
            latest[code] = max(latest.get(code, state["season"]), state["season"])
        teams = {code: [] for code in COMPETITIONS if code in latest}
        for state in self.team_states.values():
            if state["season"] == latest[state["competition"]]:
                teams[state["competition"]].append(state["team"])
        return {code: sorted(names) for code, names in teams.items()}

    def match_features(self, home_team, away_team):
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")
        for team in (home_team, away_team):
            if team not in self.team_states:
                raise ValueError(f"Unknown team: {team}")
        return match_features(self.team_states[home_team], self.team_states[away_team])

    def predict_match(self, home_team, away_team) -> dict:
        features = self.match_features(home_team, away_team)
        return self.predict([features[name] for name in self.feature_columns])

    def predict(self, features: List[float]) -> dict:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}")

        # Same computation as LogisticRegression.predict_proba (multinomial softmax)
        x = (np.array(features, dtype=float) - self.mean) / self.scale
        logits = self.coef @ x + self.intercept
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        best = int(np.argmax(probabilities))

        return {
            "prediction": LABELS[self.classes[best]],
            "confidence": float(probabilities[best]),
            "probabilities": {LABELS[c]: float(p) for c, p in zip(self.classes, probabilities)},
        }
