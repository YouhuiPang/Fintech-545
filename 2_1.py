import pandas as pd

df = pd.read_csv("test2.csv")
cols = list(df.columns)
ew_cov = df.ewm(alpha=1-0.97).cov(bias=True, pairwise=True).iloc[-len(cols):]
ew_cov.to_csv("out_2.1.csv", index=False)
