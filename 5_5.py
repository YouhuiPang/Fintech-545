import pandas as pd
import numpy as np


np.random.seed(42)
S = pd.read_csv("test5_2.csv").values
S = (S + S.T)/2
w, V = np.linalg.eigh(S)
idx = np.argsort(w)[::-1]
w = w[idx]; V = V[:, idx]
cum = np.cumsum(w)/w.sum()
k = int(np.searchsorted(cum, 0.99) + 1)
Vk = V[:, :k]; wk = w[:k]
Lk = Vk @ np.diag(np.sqrt(wk))
Z = np.random.randn(100000, k)
X = Z @ Lk.T
Cov = np.cov(X, rowvar=False, bias=False)

pd.DataFrame(Cov, columns=["x1","x2","x3","x4","x5"]).to_csv("out_5.5.csv", index=False)
