import pandas as pd
import numpy as np

np.random.seed(7)
S = pd.read_csv("test5_1.csv").values
S = (S + S.T)/2
L = np.linalg.cholesky(S)
Z = np.random.randn(100000, S.shape[0])
X = Z @ L.T
Cov = np.cov(X, rowvar=False, bias=False)
pd.DataFrame(Cov, columns=["x1","x2","x3","x4","x5"]).to_csv("out_5.1.csv", index=False)
