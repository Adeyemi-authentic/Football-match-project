"""Train the match outcome model, compare it with alternatives on recent matches and save it."""
import copy
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import COMBINED_CSV, TABLE_NAME, get_sqlalchemy_engine
from features import FEATURE_COLUMNS, LABELS, build_features
from model import MODEL_PATH, save_artifact

RAW_COLUMNS = ["id", "competition", "season", "matchday", "date",
               "home_team", "away_team", "home_score", "away_score"]

# Logistic regression (the saved model): inverse regularization strengths tried on the validation set
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# Gradient boosting (comparison only): settings tried on the validation set
GB_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "min_samples_leaf": [50, 100],
    "max_iter": [25, 50, 100, 200],
}

# Neural network (comparison only)
NN_HYPERPARAMS = {"hidden_size": 42, "hidden_size2": 18, "dropout_prob": 0.35, "output_dim": 3}
LEARNING_RATE = 0.008
MAX_EPOCHS = 2000
PATIENCE = 100           # stop when validation loss hasn't improved for this many epochs
USE_CLASS_WEIGHTS = True  # improves draw recall at some cost to overall accuracy
TEST_SHARE = 0.2         # most recent matches held out for the final evaluation
VAL_SHARE = 0.15         # most recent part of the training data, used for early stopping


def load_matches():
    """Prefer the CSV written by the scraper in this run; fall back to the database."""
    if COMBINED_CSV.exists():
        print(f"Loading data from {COMBINED_CSV}")
        data = pd.read_csv(COMBINED_CSV)
    else:
        print(f"Loading data from database table '{TABLE_NAME}'")
        data = pd.read_sql(f"SELECT * FROM {TABLE_NAME};", get_sqlalchemy_engine())
    return data[RAW_COLUMNS]


class FootballML(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_dim, dropout_prob=0.35):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def tune(make_model, candidates, X_train, y_train, X_val, y_val):
    """Return the candidate settings with the lowest validation log loss."""
    def val_loss(params):
        model = make_model(**params).fit(X_train, y_train)
        return log_loss(y_val, model.predict_proba(X_val), labels=list(LABELS))
    return min(candidates, key=val_loss)


def fit_nn(X_train, y_train, X_val=None, y_val=None, epochs=MAX_EPOCHS):
    """Full-batch training. With validation data, keeps the weights from the best epoch."""
    torch.manual_seed(42)
    model = FootballML(len(FEATURE_COLUMNS), **NN_HYPERPARAMS)
    weights = None
    if USE_CLASS_WEIGHTS:
        weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train),
                               dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_tensor, y_train_tensor = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    if X_val is not None:
        X_val_tensor, y_val_tensor = torch.FloatTensor(X_val), torch.LongTensor(y_val)

    best_loss, best_epoch, best_state = float("inf"), epochs, None
    for epoch in range(epochs):
        model.train()
        loss = criterion(model(X_train_tensor), y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if X_val is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = val_criterion(model(X_val_tensor), y_val_tensor).item()
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch + 1, copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= PATIENCE:
            break
        if (epoch + 1) % 100 == 0:
            print(f"Epoch[{epoch + 1}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def nn_predict_proba(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.FloatTensor(X)), dim=1).numpy()


def report(name, y_true, proba, with_log_loss=True):
    acc = accuracy_score(y_true, proba.argmax(axis=1))
    result = {"accuracy": round(float(acc), 4)}
    line = f"{name:<28} accuracy {acc:.3f}"
    if with_log_loss:
        loss = log_loss(y_true, proba, labels=list(LABELS))
        result["log_loss"] = round(float(loss), 4)
        line += f"   log loss {loss:.3f}"
    print(line)
    return result


def main():
    data, team_states = build_features(load_matches())
    X = data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = data["Result"].to_numpy(dtype=np.int64)

    # Time-based split: train on older matches, test on the most recent ones.
    # Settings are chosen on the validation slice; the test set is only used for the final report.
    test_start = int(len(data) * (1 - TEST_SHARE))
    val_start = int(test_start * (1 - VAL_SHARE))
    X_fit, y_fit = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_train, y_train = X[:test_start], y[:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    print(f"Train {len(X_fit)}, validation {len(X_val)}, test {len(X_test)} "
          f"(test from {data['date'].iloc[test_start].date()})")

    def logreg(C):
        return LogisticRegression(C=C, max_iter=2000)

    sc_fit = StandardScaler().fit(X_fit)
    best_C = tune(logreg, [{"C": C} for C in C_GRID],
                  sc_fit.transform(X_fit), y_fit, sc_fit.transform(X_val), y_val)["C"]
    gb_candidates = [dict(zip(GB_GRID, values)) for values in itertools.product(*GB_GRID.values())]
    gb_params = tune(lambda **p: HistGradientBoostingClassifier(**p, early_stopping=False, random_state=42),
                     gb_candidates, X_fit, y_fit, X_val, y_val)
    nn_model, best_epoch = fit_nn(sc_fit.transform(X_fit).astype(np.float32), y_fit,
                                  sc_fit.transform(X_val).astype(np.float32), y_val)
    print(f"Logistic regression C={best_C}; gradient boosting {gb_params}; neural network best epoch {best_epoch}")

    # Refit on train + validation and compare everything on the same held-out matches
    sc_train = StandardScaler().fit(X_train)
    lr_model = logreg(best_C).fit(sc_train.transform(X_train), y_train)
    gb_model = HistGradientBoostingClassifier(**gb_params, early_stopping=False, random_state=42).fit(X_train, y_train)
    train_dist = np.bincount(y_train, minlength=3) / len(y_train)
    lr_proba = lr_model.predict_proba(sc_train.transform(X_test))

    print("\nTest set results:")
    metrics = {
        "always_home_win": report("Always predict home win", y_test, np.tile([0, 0, 1.0], (len(y_test), 1)),
                                  with_log_loss=False),
        "class_frequencies": report("Training class frequencies", y_test, np.tile(train_dist, (len(y_test), 1))),
        "logistic_regression": report("Logistic regression (saved)", y_test, lr_proba),
        "gradient_boosting": report("Gradient boosting", y_test, gb_model.predict_proba(X_test)),
        "neural_network": report("Neural network", y_test,
                                 nn_predict_proba(nn_model, sc_fit.transform(X_test).astype(np.float32))),
    }

    y_pred = lr_proba.argmax(axis=1)
    print("\nLogistic regression:")
    print(classification_report(y_test, y_pred, labels=list(LABELS),
                                target_names=list(LABELS.values()), zero_division=0))
    print(confusion_matrix(y_test, y_pred, labels=list(LABELS)))

    # Final model: refit the logistic regression on all matches
    sc_final = StandardScaler().fit(X)
    final_model = logreg(best_C).fit(sc_final.transform(X), y)

    data_as_of = str(data["date"].max().date())
    save_artifact(MODEL_PATH, final_model, sc_final, team_states, data_as_of, metrics)
    print(f"Saved model to {MODEL_PATH} (data up to {data_as_of})")


if __name__ == "__main__":
    main()
