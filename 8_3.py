import numpy as np
import pandas as pd
from scipy.special import gammaln, softplus
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import t


x = pd.read_csv("test7_2.csv")["x1"].dropna().to_numpy(float)
n = x.size

class TFit(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        mu, log_sigma, raw_nu = params
        sigma = np.exp(log_sigma)
        nu = 2.0 + softplus(raw_nu)
        eps = 1e-12
        z = (x - mu) / (sigma + eps)
        c = (gammaln((nu+1)/2) - gammaln(nu/2)
             - 0.5*np.log(nu*np.pi) - np.log(sigma + eps))
        ll = c - 0.5*(nu+1)*np.log1p((z*z)/(nu + eps))
        return -ll

mu0 = float(np.mean(x))
sigma0 = float(np.std(x, ddof=1))
raw_nu0 = np.log(np.exp(max(6.0 - 2.0, 1e-6)) - 1.0)

start = np.array([mu0, np.log(max(sigma0, 1e-8)), raw_nu0], dtype=float)

mod = TFit(x)
res = mod.fit(start_params=start, method="bfgs", maxiter=2000, disp=True)

mu_hat, log_sigma_hat, raw_nu_hat = res.params
sigma_hat = float(np.exp(log_sigma_hat))
nu_hat = float(2.0 + softplus(raw_nu_hat))


np.random.seed(1)
n = 100000
alpha = 0.05
sim = t.rvs(nu_hat, loc=mu_hat, scale=sigma_hat, size=n)
k = max(0, int(np.floor(alpha*n)) - 1)
q = np.partition(sim, k)[k]

out = pd.DataFrame({
    "VaR Absolute": [abs(q)],
    "VaR Diff from Mean": [-t.ppf(alpha, nu_hat) * sigma_hat]
})
out.to_csv("out_8.3.csv", index=False)
