import pandas as pd

df = pd.read_csv("test2.csv")
cols = list(df.columns)
ew_corr = df.ewm(alpha=1-0.94).corr(pairwise=True).iloc[-len(cols):]
ew_corr.to_csv("out_2.2.csv", index=False)
