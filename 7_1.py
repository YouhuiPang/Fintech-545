import pandas as pd

data = pd.read_csv("test7_1.csv")["x1"].dropna().values

n = len(data)
mu = sum(data) / n

squared_diff = [(x - mu) ** 2 for x in data]
sigma = (sum(squared_diff) / (n - 1)) ** 0.5

out = pd.DataFrame({"mu": [mu], "sigma": [sigma]})
out.to_csv("out7_1.csv", index=False)

