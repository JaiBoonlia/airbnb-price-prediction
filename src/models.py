"""Baselines (Ridge, KNN), Random Forest, XGBoost and the MLP from the paper."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

import config


def get_sklearn_model(name: str):
    """Default sklearn/xgboost settings, as in Section 6.1 of the paper."""
    if name == "linear":          # linear regression with L2 regularisation
        return Ridge()
    if name == "knn":             # unweighted KNN, Euclidean distance
        return KNeighborsRegressor(weights="uniform")
    if name == "rf":
        return RandomForestRegressor(n_jobs=-1, random_state=config.SEED)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_jobs=-1, random_state=config.SEED)
    raise ValueError(name)


class MLPRegressor:
    """3 fully-connected layers: x -> ReLU(W1x+b1) -> ReLU(W2.+b2) -> W3.+b3.

    Hidden sizes (64, 32), SGD + Nesterov momentum (lr=0.005, mu=0.9), L2=0.005,
    MSE loss, early stopping on a 10% hold-out of the training data
    (Section 6.2 of the paper).
    """

    def __init__(self, hidden=config.NN_HIDDEN, seed: int = config.SEED):
        self.hidden, self.seed = hidden, seed

    def _build(self, d_in):
        import torch.nn as nn
        h1, h2 = self.hidden
        return nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        idx = rng.permutation(len(X))
        n_val = max(1, int(config.NN_ES_VAL_FRAC * len(X)))
        va, tr = idx[:n_val], idx[n_val:]
        Xt = torch.tensor(X[tr], dtype=torch.float32, device=self.device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=self.device).unsqueeze(1)
        Xv = torch.tensor(X[va], dtype=torch.float32, device=self.device)
        yv = y[va]

        self.net = self._build(X.shape[1]).to(self.device)
        opt = torch.optim.SGD(self.net.parameters(), lr=config.NN_LR, momentum=config.NN_MOMENTUM,
                              nesterov=True, weight_decay=config.NN_L2)
        loss_fn = torch.nn.MSELoss()

        prev_r2, stale, self.history = None, 0, []
        for epoch in range(config.NN_MAX_EPOCHS):
            self.net.train()
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(perm), config.NN_BATCH):
                b = perm[i:i + config.NN_BATCH]
                opt.zero_grad()
                loss_fn(self.net(Xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_r2 = r2_score(yv, self.net(Xv).squeeze(1).cpu().numpy())
            self.history.append(val_r2)
            # early stopping: change in validation R^2 below tol for `patience` iterations
            if prev_r2 is not None and abs(val_r2 - prev_r2) < config.NN_ES_TOL:
                stale += 1
            else:
                stale = 0
            prev_r2 = val_r2
            if stale >= config.NN_ES_PATIENCE:
                break
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32, device=self.device))
        return out.squeeze(1).cpu().numpy()


def get_model(name: str):
    return MLPRegressor() if name == "nn" else get_sklearn_model(name)


def evaluate(y_true, y_pred) -> dict:
    return {"mse": float(mean_squared_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}
