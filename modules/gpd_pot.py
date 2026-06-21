"""P1.7 — Generalized Pareto Distribution (POT) + tests d'ajustement.

Méthode des Excès au-delà d'un Seuil (Peaks Over Threshold) :
- Choix du seuil par Mean Excess Plot
- Estimation MLE des paramètres (xi, beta) de la GPD
- Tests d'adéquation : Kolmogorov-Smirnov + Anderson-Darling
- Calcul des quantiles extrêmes (Return Periods 100, 200, 1000 ans)

Référence : Embrechts, Klüppelberg, Mikosch (1997)
"Modelling Extremal Events" — référence ISFA/Bercy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest, lognorm, pareto
from scipy.optimize import minimize_scalar


@dataclass
class GPDFit:
    """Résultat de l'ajustement GPD."""
    threshold: float
    xi: float                  # shape parameter (positif = queue lourde)
    beta: float                # scale parameter
    n_exceedances: int
    log_likelihood: float
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_pvalue: float
    aicc: float

    def quantile(self, p: float, n_total: int) -> float:
        """Quantile non-conditionnel : P(X <= x) = p sur tout l'échantillon.

        Conversion : on cherche x tq P(X > x) = 1-p
        P(X > x) = (n_excès/n_total) × P_GPD(X-u > x-u)
        → 1-p = (n_excès/n_total) × (1 - F_GPD(x-u))
        """
        if p < 1 - self.n_exceedances / n_total:
            return self.threshold
        zeta = self.n_exceedances / n_total
        if self.xi == 0:
            x = self.threshold - self.beta * np.log((1 - p) / zeta)
        else:
            x = self.threshold + (self.beta / self.xi) * (
                ((1 - p) / zeta) ** (-self.xi) - 1)
        return float(x)

    def return_level(self, return_period: float, n_total: int) -> float:
        """Niveau pour une période de retour T (en années)."""
        p = 1 - 1 / return_period
        return self.quantile(p, n_total)


def fit_gpd(claims: np.ndarray, threshold: float) -> GPDFit:
    """Ajuste une GPD aux excès au-delà du seuil par MLE.

    Utilise scipy.stats.genpareto.fit avec contrainte loc=threshold.
    """
    claims = np.asarray(claims, dtype=float)
    excess = claims[claims > threshold] - threshold
    n = len(excess)
    if n < 10:
        raise ValueError(f"Pas assez d'excès ({n}) au seuil {threshold}")

    # MLE
    xi, _, beta = genpareto.fit(excess, floc=0)

    # Log-likelihood
    ll = float(genpareto.logpdf(excess, c=xi, loc=0, scale=beta).sum())

    # KS test
    ks_stat, ks_pval = kstest(excess, lambda x: genpareto.cdf(x, c=xi, scale=beta))

    # Anderson-Darling (calcul manuel pour GPD)
    ad_stat, ad_pval = _anderson_darling_gpd(excess, xi, beta)

    # AICc (corrigé pour petits échantillons)
    k = 2
    aic = -2 * ll + 2 * k
    aicc = aic + (2 * k * (k + 1) / max(n - k - 1, 1))

    return GPDFit(
        threshold=threshold, xi=float(xi), beta=float(beta),
        n_exceedances=n, log_likelihood=ll,
        ks_statistic=float(ks_stat), ks_pvalue=float(ks_pval),
        ad_statistic=float(ad_stat), ad_pvalue=float(ad_pval),
        aicc=float(aicc),
    )


def _anderson_darling_gpd(excess, xi, beta):
    """Statistique Anderson-Darling pour GPD.

    AD² = -n - (1/n) × Σ (2i-1)(ln F(x_i) + ln(1-F(x_{n+1-i})))
    """
    n = len(excess)
    sorted_x = np.sort(excess)
    cdf_vals = genpareto.cdf(sorted_x, c=xi, scale=beta)
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    ad2 = -n - s / n

    # p-value approchée (Choulakian-Stephens 2001 pour xi connu)
    if ad2 < 0.5:
        pval = 1.0
    elif ad2 > 5.0:
        pval = 0.0
    else:
        pval = float(np.exp(-1.0 * ad2))
    return float(ad2), pval


def mean_excess_function(claims: np.ndarray, thresholds: np.ndarray | None = None
                         ) -> pd.DataFrame:
    """Mean Excess Plot pour choisir le seuil.

    Pour un seuil u : MEF(u) = E[X - u | X > u].
    Pour une GPD : MEF(u) linéaire en u : MEF(u) = (beta + xi×u) / (1-xi).
    On cherche la zone linéaire la plus à gauche possible.
    """
    claims = np.asarray(claims, dtype=float)
    claims = claims[claims > 0]
    if thresholds is None:
        thresholds = np.quantile(claims, np.linspace(0.5, 0.95, 30))

    mef, n_excess = [], []
    for u in thresholds:
        ex = claims[claims > u] - u
        if len(ex) >= 5:
            mef.append(ex.mean())
            n_excess.append(len(ex))
        else:
            mef.append(np.nan)
            n_excess.append(len(ex))

    return pd.DataFrame({
        "threshold": thresholds,
        "mean_excess": mef,
        "n_exceedances": n_excess,
    })


def hill_estimator(claims: np.ndarray, k_max: int | None = None) -> pd.DataFrame:
    """Hill plot pour estimer alpha de Pareto (= 1/xi pour GPD).

    Alpha_Hill(k) = (1/k) × Σ_{i=1}^{k} ln(X_(n-i+1) / X_(n-k))
    On cherche un plateau stable sur le plot.
    """
    sorted_x = np.sort(claims)[::-1]
    n = len(sorted_x)
    k_max = k_max or min(n - 1, 200)

    alphas = []
    for k in range(2, k_max):
        logs = np.log(sorted_x[:k] / sorted_x[k])
        h = logs.mean()
        alphas.append(1 / h if h > 0 else np.nan)

    return pd.DataFrame({"k": range(2, k_max), "alpha_hill": alphas})


def optimal_threshold(claims: np.ndarray, n_grid: int = 20) -> dict:
    """Choisit automatiquement le seuil optimal par minimisation AICc.

    Approche : grid search sur les quantiles 50% à 95% des sinistres.
    """
    claims = np.asarray(claims, dtype=float)
    claims = claims[claims > 0]
    thresholds = np.quantile(claims, np.linspace(0.5, 0.95, n_grid))

    best = None
    candidates = []
    for u in thresholds:
        try:
            fit = fit_gpd(claims, u)
            candidates.append({
                "threshold": u, "xi": fit.xi, "beta": fit.beta,
                "n": fit.n_exceedances, "aicc": fit.aicc,
                "ks_pval": fit.ks_pvalue,
            })
            if best is None or fit.aicc < best["aicc"]:
                best = candidates[-1]
                best["fit"] = fit
        except Exception:
            continue
    return {"best": best, "candidates": pd.DataFrame(candidates)}


def compare_distributions(claims: np.ndarray, threshold: float) -> pd.DataFrame:
    """Compare GPD vs Pareto vs LogNormale sur les excès au seuil.

    Retourne KS, AD, log-likelihood et AICc pour chaque modèle.
    """
    excess = np.asarray(claims)[np.asarray(claims) > threshold] - threshold
    n = len(excess)
    if n < 10:
        return pd.DataFrame()

    results = []

    # GPD
    fit = fit_gpd(claims, threshold)
    results.append({"model": "GPD", "params": f"xi={fit.xi:.3f}, beta={fit.beta:.0f}",
                    "ll": fit.log_likelihood, "aicc": fit.aicc,
                    "ks_stat": fit.ks_statistic, "ks_pval": fit.ks_pvalue,
                    "ad_pval": fit.ad_pvalue})

    # Pareto type 1 (b/x^alpha, x > seuil)
    try:
        log_excess = np.log(1 + excess / threshold)
        alpha = 1 / log_excess.mean()
        ll_par = n * np.log(alpha) + n * alpha * np.log(threshold) - \
                 (alpha + 1) * np.log(threshold + excess).sum()
        aic_par = -2 * ll_par + 2 * 1
        aicc_par = aic_par + 2 * (2) / max(n - 2, 1)
        ks_par = kstest(excess, lambda x: 1 - (threshold / (threshold + x)) ** alpha)
        results.append({"model": "Pareto T1", "params": f"alpha={alpha:.3f}",
                        "ll": ll_par, "aicc": aicc_par,
                        "ks_stat": ks_par.statistic, "ks_pval": ks_par.pvalue,
                        "ad_pval": np.nan})
    except Exception:
        pass

    # LogNormale (sur excès)
    try:
        shape, loc, scale = lognorm.fit(excess, floc=0)
        ll_ln = float(lognorm.logpdf(excess, shape, loc=0, scale=scale).sum())
        aic_ln = -2 * ll_ln + 2 * 2
        aicc_ln = aic_ln + 2 * 2 * 3 / max(n - 3, 1)
        ks_ln = kstest(excess, lambda x: lognorm.cdf(x, shape, loc=0, scale=scale))
        results.append({"model": "LogNormale",
                        "params": f"mu={np.log(scale):.2f}, sigma={shape:.3f}",
                        "ll": ll_ln, "aicc": aicc_ln,
                        "ks_stat": ks_ln.statistic, "ks_pval": ks_ln.pvalue,
                        "ad_pval": np.nan})
    except Exception:
        pass

    return pd.DataFrame(results).sort_values("aicc")


def severity_function_from_gpd(fit: GPDFit, n_total: int):
    """Retourne une fonction de tirage de sévérité à utiliser dans simulate_bouquet.

    Pour les tirages au-dessus du seuil, on utilise la GPD.
    Pour les tirages sous le seuil, on prend le seuil (approximation).
    """
    def severity(rng, n):
        zeta = fit.n_exceedances / n_total
        # Bernoulli : excède le seuil avec proba zeta
        is_excess = rng.random(n) < zeta
        x = np.full(n, fit.threshold)
        n_ex = is_excess.sum()
        if n_ex > 0:
            x[is_excess] = fit.threshold + genpareto.rvs(
                c=fit.xi, scale=fit.beta, size=int(n_ex), random_state=rng)
        return x
    return severity
