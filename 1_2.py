import pandas as pd

df = pd.read_csv("test1.csv")
df_skip = df.dropna()
corr_matrix = df_skip.corr()
corr_matrix.to_csv("out_1.2.csv", index=False)
