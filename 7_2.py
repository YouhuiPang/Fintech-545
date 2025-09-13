import pandas as pd
from scipy.stats import t

x = pd.read_csv("test7_2.csv")["x1"].dropna().values

nu, mu, sigma = t.fit(x)

out = pd.DataFrame({"mu": [mu], "sigma": [sigma], "nu": [nu]})
out.to_csv("out7_2.csv", index=False)
