import pandas as pd
import numpy as np


def higham_cov(S, max_iter=200, tol=1e-10):
  S = (S + S.T) / 2
  std = np.sqrt(np.diag(S))
  Dinv = np.diag(1 / std)
  R = Dinv @ S @ Dinv
  X = (R + R.T) / 2
  Y = np.zeros_like(X)
  for _ in range(max_iter):
    Sx = X - Y
    w, V = np.linalg.eigh((Sx + Sx.T) / 2)
    w = np.clip(w, 0, None)
    X_psd = V @ np.diag(w) @ V.T
    Y = X_psd - Sx
    Xn = X_psd.copy()
    np.fill_diagonal(Xn, 1.0)
    if np.linalg.norm(Xn - X, "fro") < tol * max(1.0, np.linalg.norm(X, "fro")):
      X = Xn;
      break
    X = Xn
  S_fix = np.diag(std) @ X @ np.diag(std)
  return (S_fix + S_fix.T) / 2


np.random.seed(42)
S0 = pd.read_csv("test5_3.csv").values
S = higham_cov(S0)
w, V = np.linalg.eigh((S + S.T) / 2)
w = np.clip(w, 0.0, None)
L = V @ np.diag(np.sqrt(w))
Z = np.random.randn(100000, S.shape[0])
X = Z @ L.T
Cov = np.cov(X, rowvar=False, bias=False)
pd.DataFrame(Cov, columns=["x1", "x2", "x3", "x4", "x5"]).to_csv("out_5.4.csv", index=False)
