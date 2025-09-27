import pandas as pd

df = pd.read_csv("test1.csv")
cov_matrix = df.cov(min_periods=1)
cov_matrix.to_csv("out_1.3.csv", index=False)
