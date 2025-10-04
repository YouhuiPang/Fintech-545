import pandas as pd
import numpy as np


def near_psd_cov(S):
  S = (S + S.T) / 2
  std = np.sqrt(np.diag(S))
  C = np.diag(1 / std) @ S @ np.diag(1 / std)
  w, V = np.linalg.eigh((C + C.T) / 2)
  w = np.clip(w, 0, None)
  C = V @ np.diag(w) @ V.T
  s = np.sqrt(1 / np.diag(C))
  C = (C * s).T * s
  return np.diag(std) @ C @ np.diag(std)


np.random.seed(42)
S0 = pd.read_csv("test5_3.csv").values
S = near_psd_cov(S0)
S = (S + S.T) / 2


w, V = np.linalg.eigh(S)
w = np.clip(w, 0, None)
L = V @ np.diag(np.sqrt(w))

Z = np.random.randn(100000, S.shape[0])
X = Z @ L.T
Cov = np.cov(X, rowvar=False, bias=False)

pd.DataFrame(Cov, columns=["x1", "x2", "x3", "x4", "x5"]).to_csv("out_5.3.csv", index=False)
