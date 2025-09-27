import numpy as np
import pandas as pd


def higham_corr(corr_df: pd.DataFrame, max_iter: int = 200, tol: float = 1e-10) -> pd.DataFrame:
    X = (corr_df.values + corr_df.values.T) / 2
    Y = np.zeros_like(X)
    for _ in range(max_iter):
        R = X - Y
        w, V = np.linalg.eigh((R + R.T) / 2)
        w[w < 0] = 0
        X_psd = V @ np.diag(w) @ V.T
        Y = X_psd - R
        X_new = X_psd.copy()
        np.fill_diagonal(X_new, 1.0)
        if np.linalg.norm(X_new - X, ord='fro') < tol * max(1.0, np.linalg.norm(X, ord='fro')):
            X = X_new
            break
        X = X_new
    X = (X + X.T) / 2
    return pd.DataFrame(X, columns=corr_df.columns)


corr = pd.read_csv("out_1.4.csv")
corr_higham = higham_corr(corr)
corr_higham.to_csv("out_3.4.csv", index=False)
