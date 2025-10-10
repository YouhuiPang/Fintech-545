import pandas as pd
import numpy as np
from scipy.stats import norm

r = pd.read_csv("test7_1.csv")["x1"].to_numpy(float)
alpha = 0.05
mu = r.mean()
sigma = r.std(ddof=1)

z = norm.ppf(alpha)
phi = norm.pdf(z)

ES_diff = sigma * (phi / alpha)
ES_abs  = - (mu - ES_diff)

pd.DataFrame({"ES Absolute":[ES_abs], "ES Diff from Mean":[ES_diff]}).to_csv("out_8.4.csv", index=False)
