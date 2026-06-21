"""P2.1 — Copules pour dépendance multi-branches.

Famille de copules implémentées :
- Gaussienne (elliptique, dépendance symétrique linéaire)
- t-Student (elliptique, dépendance de queue symétrique)
- Clayton (archimédienne, dépendance de queue inférieure)
- Gumbel (archimédienne, dépendance de queue supérieure — cas Cat)

Utilisation typique : modéliser la corrélation des LR entre LOB
(Fire / Engineering / GA / Motor) pour des scénarios bouquet réalistes.

Référence : Nelsen (2006) "An Introduction to Copulas".
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, kendalltau, rankdata


@dataclass
class CopulaFit:
    family: str
    params: dict
    kendall_tau: np.ndarray   # matrice de tau de Kendall par paire
    log_likelihood: float
    aic: float
    n_obs: int


def empirical_uniform(data: np.ndarray) -> np.ndarray:
    """Transforme un échantillon multivarié en pseudo-uniformes par rang.

    Pour chaque colonne : u_i = rank(x_i) / (n+1) ∈ (0,1).
    Évite la mauvaise spécification des marginales (approche non-paramétrique).
    """
    n = len(data)
    return np.column_stack([rankdata(data[:, j]) / (n + 1)
                             for j in range(data.shape[1])])


def fit_gaussian_copula(u: np.ndarray) -> CopulaFit:
    """Calibre une copule Gaussienne.

    Méthode : on transforme u_i en z_i = Φ^{-1}(u_i), puis la matrice de
    corrélation R des z_i est l'estimateur ML.
    """
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    R = np.corrcoef(z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)  # colonnes à variance nulle → corr=0
    np.fill_diagonal(R, 1.0)
    n, d = u.shape

    sign, logdet = np.linalg.slogdet(R)
    R_inv = np.linalg.inv(R)
    quad = np.einsum('ij,jk,ik->i', z, R_inv - np.eye(d), z)
    ll = -0.5 * (n * logdet + quad.sum())

    k = d * (d - 1) / 2
    aic = -2 * ll + 2 * k

    # Tau de Kendall théorique pour copule gaussienne : τ = 2/π × arcsin(ρ)
    tau = (2 / np.pi) * np.arcsin(R)
    return CopulaFit("gaussian", {"R": R}, tau, float(ll), float(aic), n)


def fit_t_copula(u: np.ndarray, nu_grid: list[float] | None = None) -> CopulaFit:
    """Calibre une copule t-Student (recherche du nu optimal).

    nu petit → dépendance de queue forte, nu grand → tend vers gaussienne.
    """
    if nu_grid is None:
        nu_grid = [3, 4, 5, 8, 12, 20, 30]
    best = None
    n, d = u.shape

    for nu in nu_grid:
        z = student_t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)
        R = np.corrcoef(z, rowvar=False)
        sign, logdet = np.linalg.slogdet(R)
        R_inv = np.linalg.inv(R)
        quad = np.einsum('ij,jk,ik->i', z, R_inv, z)
        ll_t = (
            n * (np.log(np.math.gamma((nu + d) / 2))
                 - np.log(np.math.gamma(nu / 2))
                 - (d - 1) * np.log(np.math.gamma((nu + 1) / 2))
                 + (d - 1) * np.log(np.math.gamma(nu / 2)))
            - 0.5 * n * logdet
            - 0.5 * (nu + d) * np.log1p(quad / nu).sum()
            + 0.5 * (nu + 1) * np.log1p(z ** 2 / nu).sum()
        )
        if best is None or ll_t > best['ll']:
            tau = (2 / np.pi) * np.arcsin(R)
            best = {"nu": nu, "R": R, "tau": tau, "ll": ll_t}

    k = d * (d - 1) / 2 + 1
    aic = -2 * best['ll'] + 2 * k
    return CopulaFit("t-student", {"R": best["R"], "nu": best["nu"]},
                     best["tau"], float(best["ll"]), float(aic), n)


def fit_clayton_copula(u: np.ndarray) -> CopulaFit:
    """Calibre une copule Clayton bivariée par tau de Kendall.

    Clayton : θ = 2τ / (1−τ)
    Capture la dépendance de queue inférieure (sinistres simultanés).
    Pour d > 2 on retourne theta moyen.
    """
    n, d = u.shape
    taus = np.zeros((d, d))
    thetas = []
    for i in range(d):
        for j in range(d):
            if i == j:
                taus[i, j] = 1.0
                continue
            tau, _ = kendalltau(u[:, i], u[:, j])
            taus[i, j] = tau
            if i < j and tau < 1:
                thetas.append(max(2 * tau / (1 - tau), 0))
    theta = np.mean(thetas) if thetas else 0

    # Log-likelihood approchée pour Clayton bivariée
    ll = 0
    for i in range(d):
        for j in range(i + 1, d):
            ll += _clayton_logpdf_bivariate(u[:, i], u[:, j], theta).sum()

    return CopulaFit("clayton", {"theta": theta}, taus,
                     float(ll), float(-2 * ll + 2), n)


def fit_gumbel_copula(u: np.ndarray) -> CopulaFit:
    """Calibre une copule Gumbel bivariée par tau de Kendall.

    Gumbel : θ = 1 / (1−τ), θ ≥ 1.
    Capture la dépendance de queue supérieure (utile pour Cat XL).
    """
    n, d = u.shape
    taus = np.zeros((d, d))
    thetas = []
    for i in range(d):
        for j in range(d):
            if i == j:
                taus[i, j] = 1.0
                continue
            tau, _ = kendalltau(u[:, i], u[:, j])
            taus[i, j] = tau
            if i < j and tau < 1 and tau > 0:
                thetas.append(1 / (1 - tau))
    theta = np.mean(thetas) if thetas else 1.0
    return CopulaFit("gumbel", {"theta": theta}, taus, 0.0, 0.0, n)


def _clayton_logpdf_bivariate(u, v, theta):
    """Densité Clayton bivariée."""
    if theta <= 0:
        return np.zeros_like(u)
    return (np.log(1 + theta)
            + (-1 - theta) * (np.log(u) + np.log(v))
            + (-2 - 1 / theta) * np.log(u ** (-theta) + v ** (-theta) - 1))


def sample_gaussian_copula(rng: np.random.Generator, R: np.ndarray, n: int
                            ) -> np.ndarray:
    """Tire n observations multivariées d'une copule Gaussienne."""
    d = R.shape[0]
    z = rng.multivariate_normal(np.zeros(d), R, size=n)
    return norm.cdf(z)


def sample_t_copula(rng: np.random.Generator, R: np.ndarray, nu: float, n: int
                     ) -> np.ndarray:
    """Tire n observations d'une copule t-Student."""
    d = R.shape[0]
    z = rng.multivariate_normal(np.zeros(d), R, size=n)
    chi = rng.chisquare(nu, size=n) / nu
    t_vals = z / np.sqrt(chi[:, None])
    return student_t.cdf(t_vals, df=nu)


def sample_clayton_copula(rng: np.random.Generator, theta: float, n: int, d: int
                           ) -> np.ndarray:
    """Tire n observations d'une copule Clayton via Marshall-Olkin.

    Limité aux cas où theta > 0 (sinon = indépendance).
    """
    if theta <= 0:
        return rng.uniform(size=(n, d))
    # Algorithme de Marshall-Olkin : V ~ Gamma(1/θ, 1), U_i = (1 - log(W_i)/V)^{-1/θ}
    V = rng.gamma(1 / theta, 1, size=n)
    W = rng.uniform(size=(n, d))
    U = (1 - np.log(W) / V[:, None]) ** (-1 / theta)
    return U


def best_copula(data: np.ndarray, families: list[str] = None) -> CopulaFit:
    """Sélectionne la meilleure copule par AIC."""
    u = empirical_uniform(data)
    families = families or ["gaussian", "t-student", "clayton"]
    fits = []
    for fam in families:
        try:
            if fam == "gaussian":
                fits.append(fit_gaussian_copula(u))
            elif fam == "t-student":
                fits.append(fit_t_copula(u))
            elif fam == "clayton":
                fits.append(fit_clayton_copula(u))
            elif fam == "gumbel":
                fits.append(fit_gumbel_copula(u))
        except Exception:
            continue
    return min(fits, key=lambda f: f.aic)


def aggregate_lr_with_copula(
    rng: np.random.Generator,
    lr_marginals: list[tuple[float, float]],  # [(mean, std), ...]
    copula_fit: CopulaFit,
    premiums: list[float],
    n_sim: int = 10000,
) -> dict:
    """Simule des LR par LOB avec copule + calcule charge bouquet diversifiée.

    Étapes :
      1. Tire u_ij ∈ (0,1)^d depuis la copule
      2. Transforme en LR via LR_marginal_i^{-1}(u_ij) (loi LogNormale)
      3. Calcule charge_i = LR_i × premium_i
      4. Charge bouquet = somme
    """
    d = len(lr_marginals)
    if copula_fit.family == "gaussian":
        u = sample_gaussian_copula(rng, copula_fit.params["R"], n_sim)
    elif copula_fit.family == "t-student":
        u = sample_t_copula(rng, copula_fit.params["R"],
                             copula_fit.params["nu"], n_sim)
    elif copula_fit.family == "clayton":
        u = sample_clayton_copula(rng, copula_fit.params["theta"], n_sim, d)
    else:
        u = rng.uniform(size=(n_sim, d))

    # Transformation marginale LogNormale (E=mean, std=std)
    losses = np.zeros((n_sim, d))
    for j, (m, s) in enumerate(lr_marginals):
        if m <= 0 or s <= 0:
            losses[:, j] = m * premiums[j]
            continue
        sigma2 = np.log(1 + (s / m) ** 2)
        mu = np.log(m) - sigma2 / 2
        lr = np.exp(norm.ppf(np.clip(u[:, j], 1e-6, 1 - 1e-6),
                              loc=mu, scale=np.sqrt(sigma2)))
        losses[:, j] = lr * premiums[j]

    bouquet = losses.sum(axis=1)
    sum_marginal_capital = sum(2.576 * s * p for (m, s), p in zip(lr_marginals, premiums))
    var995 = float(np.quantile(bouquet, 0.995))
    tvar99 = float(bouquet[bouquet >= np.quantile(bouquet, 0.99)].mean())
    cap_diversified = max(var995 - bouquet.mean(), 0)
    diversification_benefit = (1 - cap_diversified / sum_marginal_capital
                                if sum_marginal_capital > 0 else 0)

    return {
        "losses_per_lob": losses,
        "loss_bouquet": bouquet,
        "mean": float(bouquet.mean()),
        "std": float(bouquet.std()),
        "var_99_5": var995,
        "tvar_99": tvar99,
        "capital_s2_diversified": cap_diversified,
        "capital_s2_sum_standalone": float(sum_marginal_capital),
        "diversification_benefit_pct": float(diversification_benefit),
        "correlations_implied": float(np.corrcoef(losses, rowvar=False).mean()
                                       if d > 1 else 1.0),
    }
