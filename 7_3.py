import numpy as np
import pandas as pd
from scipy.special import gammaln, softplus
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import t as tdist

df = pd.read_csv("test7_3.csv")
y = df["y"].to_numpy(float)
X = df[["x1","x2","x3"]].to_numpy(float)
n = len(y)
X = np.column_stack([np.ones(n), X])
p = X.shape[1]

beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
resid_ols = y - X @ beta_ols
sigma0 = float(np.std(resid_ols, ddof=p))
raw_nu0 = np.log(np.exp(max(6.0 - 2.0, 1e-6)) - 1.0)

class TRegression(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        alpha, b1, b2, b3, log_sigma, raw_nu = params
        sigma = np.exp(log_sigma)
        nu = 2.0 + softplus(raw_nu)
        r = y - (alpha + b1*X[:,1] + b2*X[:,2] + b3*X[:,3])
        eps = 1e-12
        z = r / (sigma + eps)
        c = (gammaln((nu+1)/2) - gammaln(nu/2)
             - 0.5*np.log(nu*np.pi) - log_sigma)
        ll = c - 0.5*(nu+1)*np.log1p((z*z)/(nu + eps))
        return -ll

start = np.array([*beta_ols, np.log(max(sigma0,1e-8)), raw_nu0], dtype=float)

mod = TRegression(y)
res = mod.fit(start_params=start, method="bfgs", maxiter=2000, disp=True)

alpha, b1, b2, b3, log_sigma_hat, raw_nu_hat = res.params
sigma_hat = float(np.exp(log_sigma_hat))
nu_hat = float(2.0 + softplus(raw_nu_hat))
mu_hat = 0.0

err_dist = tdist(df=nu_hat, loc=0.0, scale=sigma_hat)
def eval_model(X_new, u):
    Xn = np.column_stack([np.ones(len(X_new)), X_new])
    yhat = Xn @ np.array([alpha, b1, b2, b3])
    return yhat + err_dist.ppf(u)

errors = y - eval_model(df[["x1","x2","x3"]].to_numpy(float), np.full(n, 0.5))
u_vals = err_dist.cdf(errors)

pd.DataFrame({"mu": [mu_hat], "sigma": [sigma_hat], "nu": [nu_hat], "Alpha": [alpha], "B1": [b1], "B2": [b2], "B3": [b3],}).to_csv("out7_3.csv", index=False)
