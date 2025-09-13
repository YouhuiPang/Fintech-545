import pandas as pd
import numpy as np

x = pd.read_csv("test7_1.csv")["x1"].dropna().to_numpy(float)
n = x.size

mu = float(np.sum(x) / n)
sigma = float(np.sqrt(np.sum((x - mu)**2) / (n - 1)))

out = pd.DataFrame({"mu": [mu], "sigma": [sigma]})
out.to_csv("out7_1.csv", index=False)

