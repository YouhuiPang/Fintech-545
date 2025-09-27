import pandas as pd
import numpy as np

df = pd.read_csv("test6.csv")
logret = np.log(df.drop(columns=["Date"])).diff()
out = pd.concat([df["Date"], logret], axis=1).dropna()
out.to_csv("out_6.2.csv", index=False)
