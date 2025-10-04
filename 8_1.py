import pandas as pd
import numpy as np
from scipy.stats import norm


x = pd.read_csv("test7_1.csv")["x1"].dropna().to_numpy(float)
n = x.size

mu = float(np.sum(x) / n)
sigma = float(np.sqrt(np.sum((x - mu)**2) / (n - 1)))
q = mu + sigma * norm.ppf(0.05)

out = pd.DataFrame({
    "VaR Absolute": [abs(q)],
    "VaR Diff from Mean": [-norm.ppf(0.05) * sigma]
})
out.to_csv("out_8.1.csv", index=False)
