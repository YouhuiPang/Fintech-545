import pandas as pd
import numpy as np

df = pd.read_csv("test2.csv")
cols = list(df.columns)

ew_var = df.ewm(alpha=1-0.97).var(bias=True).iloc[-1]
ew_corr = df.ewm(alpha=1-0.94).corr(pairwise=True).iloc[-len(cols):]

scale = np.sqrt(ew_var)
ew_cov = ew_corr * np.outer(scale, scale)

ew_cov.to_csv("out_2.3.csv", index=False)
