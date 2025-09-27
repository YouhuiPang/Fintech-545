import pandas as pd

df = pd.read_csv("test6.csv")
ret = df.drop(columns=["Date"]).pct_change()
out = pd.concat([df["Date"], ret], axis=1).dropna()
out.to_csv("out_6.1.csv", index=False)
