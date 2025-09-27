import pandas as pd

df = pd.read_csv("test1.csv")
df_skip = df.dropna()
cov_matrix = df_skip.cov()
cov_matrix.to_csv("out_1.1.csv", index=False)
