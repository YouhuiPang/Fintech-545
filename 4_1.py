import pandas as pd
import numpy as np

A = pd.read_csv("out_3.1.csv")
L = np.linalg.cholesky(((A.values + A.values.T) / 2))
pd.DataFrame(L, columns=A.columns).to_csv("out_4.1.csv", index=False)
