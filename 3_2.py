import numpy as np
import pandas as pd


def near_psd_corr(corr_df: pd.DataFrame) -> pd.DataFrame:
    C = corr_df.values
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    w[w < 0] = 0
    C_psd = V @ np.diag(w) @ V.T
    s = np.sqrt(1 / np.diag(C_psd))
    C_psd = (C_psd * s).T * s
    np.fill_diagonal(C_psd, 1.0)
    return pd.DataFrame(C_psd, columns=corr_df.columns)


corr = pd.read_csv("out_1.4.csv")
corr_psd = near_psd_corr(corr)
corr_psd.to_csv("out_3.2.csv", index=False)
