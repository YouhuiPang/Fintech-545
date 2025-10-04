import pandas as pd
import numpy as np

np.random.seed(7)
S = pd.read_csv("test5_2.csv").values
S = (S + S.T)/2
w, V = np.linalg.eigh(S)
L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
Z = np.random.randn(100000, S.shape[0])
X = Z @ L.T
Cov = np.cov(X, rowvar=False, bias=False)
pd.DataFrame(Cov, columns=["x1","x2","x3","x4","x5"]).to_csv("out_5.2.csv", index=False)
