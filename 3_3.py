import numpy as np
import pandas as pd


def higham_cov(cov_df: pd.DataFrame, max_iter: int = 200, tol: float = 1e-10) -> pd.DataFrame:
    A = cov_df.values
    std = np.sqrt(np.diag(A))
    Dinv = np.diag(1 / std)
    R = Dinv @ A @ Dinv
    X = (R + R.T) / 2
    Y = np.zeros_like(X)
    for _ in range(max_iter):
        S = X - Y
        w, V = np.linalg.eigh((S + S.T) / 2)
        w[w < 0] = 0
        X_psd = V @ np.diag(w) @ V.T
        Y = X_psd - S
        X_new = X_psd.copy()
        np.fill_diagonal(X_new, 1.0)
        if np.linalg.norm(X_new - X, ord='fro') < tol * max(1.0, np.linalg.norm(X, ord='fro')):
            X = X_new
            break
        X = X_new
    D = np.diag(std)
    A_higham = D @ X @ D
    A_higham = (A_higham + A_higham.T) / 2
    return pd.DataFrame(A_higham, columns=cov_df.columns)


cov = pd.read_csv("out_1.3.csv")
cov_higham = higham_cov(cov)
cov_higham.to_csv("out_3.3.csv", index=False)
