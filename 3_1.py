import numpy as np
import pandas as pd


def near_psd_cov(cov_df: pd.DataFrame) -> pd.DataFrame:
    A = cov_df.values
    std = np.sqrt(np.diag(A))
    C = np.diag(1/std) @ A @ np.diag(1/std)
    w, V = np.linalg.eigh((C + C.T) / 2)
    w[w < 0] = 0
    C_psd = V @ np.diag(w) @ V.T
    s = np.sqrt(1 / np.diag(C_psd))
    C_psd = (C_psd * s).T * s
    A_psd = np.diag(std) @ C_psd @ np.diag(std)
    return pd.DataFrame(A_psd, columns=cov_df.columns)


cov = pd.read_csv("out_1.3.csv")
cov_psd = near_psd_cov(cov)
cov_psd.to_csv("out_3.1.csv", index=False)
