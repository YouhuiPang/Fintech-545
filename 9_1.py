import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.special import gammaln, softplus
from statsmodels.base.model import GenericLikelihoodModel

port = pd.read_csv("test9_1_portfolio.csv")
rets = pd.read_csv("test9_1_returns.csv")

stocks = list(port["Stock"])
vals0 = (port["Holding"] * port["Starting Price"]).to_numpy(float)
pv0 = float(vals0.sum())
rA = rets[stocks[0]].dropna().to_numpy(float)
rB = rets[stocks[1]].dropna().to_numpy(float)

alpha = 0.05
z = norm.ppf(alpha)
phi = norm.pdf(z)

muA = float(rA.mean())
sigA = float(rA.std(ddof=1))
VaR95_A_pct = -(muA + sigA * z)
ES95_A_pct  =  sigA * (phi / alpha) - muA
VaR95_A = VaR95_A_pct * float(vals0[0])
ES95_A  = ES95_A_pct  * float(vals0[0])

xB = rB.copy()

class TFit(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        mu, log_sigma, raw_nu = params
        sigma = np.exp(log_sigma)
        nu = 2.0 + softplus(raw_nu)
        eps = 1e-12
        z = (xB - mu) / (sigma + eps)
        c = (gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(nu*np.pi) - np.log(sigma + eps))
        ll = c - 0.5*(nu+1)*np.log1p((z*z)/(nu + eps))
        return -ll

mu0 = float(np.mean(xB))
sigma0 = float(np.std(xB, ddof=1))
raw_nu0 = np.log(np.exp(max(6.0 - 2.0, 1e-6)) - 1.0)
start = np.array([mu0, np.log(max(sigma0, 1e-8)), raw_nu0], dtype=float)

res = TFit(xB).fit(start_params=start, method="bfgs", maxiter=2000, disp=True)
muB, log_sigmaB, raw_nuB = res.params
sigB = float(np.exp(log_sigmaB))
nuB  = float(2.0 + softplus(raw_nuB))

a = t.ppf(alpha, nuB)
pdf_a = t.pdf(a, nuB)
VaR95_B_pct = -(muB + sigB * a)
ES95_B_pct  =  sigB * ((nuB + a*a)/((nuB - 1.0)*alpha) * pdf_a) - muB
VaR95_B = VaR95_B_pct * float(vals0[1])
ES95_B  = ES95_B_pct  * float(vals0[1])

w = vals0 / pv0
rp = rets[stocks].to_numpy(float).dot(w)
q = np.quantile(rp, alpha, method="lower")
es = rp[rp <= q].mean()
VaR95_T_pct = float(-q)
ES95_T_pct  = float(-es)
VaR95_T = VaR95_T_pct * pv0
ES95_T  = ES95_T_pct  * pv0

pd.DataFrame([
    {"Stock": stocks[0], "VaR95": VaR95_A, "ES95": ES95_A, "VaR95_Pct": VaR95_A_pct, "ES95_Pct": ES95_A_pct},
    {"Stock": stocks[1], "VaR95": VaR95_B, "ES95": ES95_B, "VaR95_Pct": VaR95_B_pct, "ES95_Pct": ES95_B_pct},
    {"Stock": "Total",   "VaR95": VaR95_T, "ES95": ES95_T, "VaR95_Pct": VaR95_T_pct, "ES95_Pct": ES95_T_pct},
]).to_csv("out_9.1.csv", index=False)
