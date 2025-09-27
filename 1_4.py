import pandas as pd

df = pd.read_csv("test1.csv")
corr_matrix = df.corr(min_periods=1)
corr_matrix.to_csv("out_1.4.csv", index=False)
