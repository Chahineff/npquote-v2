"""P2.5 — Crédibilité Bühlmann-Straub + hook PyMC pour modèles hiérarchiques.

Crédibilité Bühlmann-Straub : pondération bayésienne entre l'expérience
propre du contrat (taux observé) et l'a priori du portefeuille ou marché.

Formule centrale :
  Z = w_total / (w_total + K)
  où K = σ²_hypothèse / τ²_paramètre
  μ_crédible = Z × x̄ + (1 − Z) × μ_a_priori

Z = facteur de crédibilité ∈ [0, 1].

Référence : Bühlmann & Gisler (2005) "A Course in Credibility Theory".
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CredibilityResult:
    z_factor: float          # crédibilité ∈ [0, 1]
    mu_a_priori: float
    mu_observed: float       # moyenne pondérée des observations
    mu_credible: float       # estimateur crédible final
    sigma2_within: float     # variance intra-contrat
    tau2_between: float      # variance inter-contrats
    K: float
    n_obs: int
    total_weight: float


def buhlmann_straub(
    observations: np.ndarray,   # taux observés (par année)
    weights: np.ndarray,         # poids (= primes ou EPI)
    mu_a_priori: float | None = None,
    sigma2_within: float | None = None,
    tau2_between: float | None = None,
) -> CredibilityResult:
    """Estimateur de crédibilité Bühlmann-Straub.

    Si sigma2_within / tau2_between ne sont pas fournis, ils sont estimés
    à partir de la série (estimation hors-portefeuille requise pour
    une crédibilité véritablement bayésienne — voir buhlmann_straub_portfolio).
    """
    obs = np.asarray(observations, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(obs)
    if n == 0 or w.sum() == 0:
        return CredibilityResult(0, mu_a_priori or 0, 0, mu_a_priori or 0,
                                  0, 0, np.inf, n, 0)

    mu_obs = float(np.average(obs, weights=w))
    w_total = float(w.sum())

    if sigma2_within is None:
        if n > 1:
            sigma2_within = float(((w * (obs - mu_obs) ** 2).sum())
                                   / max(n - 1, 1))
        else:
            sigma2_within = float(np.var(obs)) if n else 0

    if tau2_between is None:
        # Approximation : si non fourni, on prend la variance externe
        # comme un quart de la variance interne (a priori conservateur).
        tau2_between = max(sigma2_within * 0.25, 1e-9)

    if mu_a_priori is None:
        mu_a_priori = mu_obs

    K = sigma2_within / tau2_between
    z = w_total / (w_total + K)
    mu_cred = z * mu_obs + (1 - z) * mu_a_priori

    return CredibilityResult(
        z_factor=float(z), mu_a_priori=float(mu_a_priori),
        mu_observed=float(mu_obs), mu_credible=float(mu_cred),
        sigma2_within=float(sigma2_within),
        tau2_between=float(tau2_between),
        K=float(K), n_obs=n, total_weight=w_total,
    )


def buhlmann_straub_portfolio(
    contracts_data: dict[str, dict],  # {contract_id: {obs, weights}}
) -> dict:
    """Bühlmann-Straub multi-contrat : estime sigma² et tau² inter-portefeuille.

    Donne une estimation portefeuille-wide des paramètres de structure
    sigma² (variance intra) et tau² (variance inter), puis applique la
    crédibilité à chaque contrat individuellement.
    """
    n_contracts = len(contracts_data)
    if n_contracts == 0:
        return {}

    # Statistiques par contrat
    contract_means = {}
    contract_weights = {}
    contract_n = {}
    for cid, data in contracts_data.items():
        obs = np.asarray(data['obs'], dtype=float)
        w = np.asarray(data['weights'], dtype=float)
        if w.sum() == 0:
            continue
        contract_means[cid] = float(np.average(obs, weights=w))
        contract_weights[cid] = float(w.sum())
        contract_n[cid] = len(obs)

    # Moyenne globale pondérée (overall a priori)
    total_w = sum(contract_weights.values())
    mu_global = sum(m * contract_weights[c] for c, m in contract_means.items()) / total_w

    # Estimation σ² (within) : moyenne pondérée des variances intra
    s2_within_sum = 0
    n_within = 0
    for cid, data in contracts_data.items():
        obs = np.asarray(data['obs'], dtype=float)
        w = np.asarray(data['weights'], dtype=float)
        if len(obs) > 1 and w.sum() > 0:
            mean_c = contract_means[cid]
            s2_within_sum += ((w * (obs - mean_c) ** 2).sum())
            n_within += (len(obs) - 1)
    sigma2 = s2_within_sum / max(n_within, 1)

    # Estimation τ² (between) selon Bühlmann-Straub :
    # τ² = (B - (n_contracts - 1) × σ²) / (total_w - Σ wᵢ²/total_w)
    B = sum(contract_weights[c] * (contract_means[c] - mu_global) ** 2
            for c in contract_means)
    denom = total_w - sum(w ** 2 for w in contract_weights.values()) / total_w
    tau2 = max((B - (n_contracts - 1) * sigma2) / max(denom, 1e-9), 1e-9)

    # Applique la crédibilité à chaque contrat
    results = {}
    for cid, data in contracts_data.items():
        if cid not in contract_means:
            continue
        results[cid] = buhlmann_straub(
            data['obs'], data['weights'],
            mu_a_priori=mu_global,
            sigma2_within=sigma2,
            tau2_between=tau2,
        )

    return {
        "mu_global_a_priori": mu_global,
        "sigma2_within_estimated": sigma2,
        "tau2_between_estimated": tau2,
        "K_global": sigma2 / tau2,
        "n_contracts": n_contracts,
        "results_by_contract": results,
    }


def credible_loss_ratio(
    contract_lrs: np.ndarray,    # LR observés par année du contrat
    contract_premiums: np.ndarray,  # primes correspondantes
    market_lr: float,
    market_variance: float | None = None,
) -> CredibilityResult:
    """Cas d'usage typique : LR crédible d'une cédante combinant son
    expérience et le benchmark marché.

    market_variance : variance des LR observés au niveau marché (tau²).
    Si None, on utilise une heuristique conservatrice.
    """
    return buhlmann_straub(
        contract_lrs, contract_premiums,
        mu_a_priori=market_lr,
        sigma2_within=None,  # estimé sur le contrat
        tau2_between=market_variance,
    )


def credibility_weighted_burning_cost(
    bc_observed: float,
    n_years_observed: int,
    bc_a_priori: float,
    full_credibility_n_years: int = 15,
) -> dict:
    """Crédibilité simplifiée pour burning cost (méthode classique américaine).

    Z = √(n / N_full) où N_full = nb d'années pour pleine crédibilité.
    Très utilisée dans le marché US (Casualty Actuarial Society).
    """
    z = min(np.sqrt(n_years_observed / full_credibility_n_years), 1.0)
    bc_credible = z * bc_observed + (1 - z) * bc_a_priori
    return {
        "z_factor": float(z),
        "bc_observed": bc_observed,
        "bc_a_priori": bc_a_priori,
        "bc_credible": float(bc_credible),
        "n_years_observed": n_years_observed,
        "full_credibility_threshold": full_credibility_n_years,
    }


def pymc_hierarchical_credibility_stub():
    """Hook pour modèle hiérarchique bayésien PyMC.

    À implémenter quand PyMC est disponible. Schéma proposé :

      with pm.Model() as model:
          mu_global ~ Normal(0.6, 0.2)       # LR global a priori
          tau ~ HalfNormal(0.1)              # dispersion inter-contrats
          mu_contract = Normal(mu_global, tau, shape=n_contracts)

          sigma ~ HalfNormal(0.05)            # dispersion intra-contrat
          obs = Normal(mu_contract[idx], sigma, observed=lr_data)

          trace = pm.sample(2000, return_inferencedata=True)

    Avantages : crédibilité, intervalles, partial pooling automatique,
                hiérarchie multi-niveau (pays > cédante > traité).
    """
    return {
        "status": "stub",
        "message": "PyMC hierarchical Bayesian credibility — install pymc to enable",
        "schema": "mu_global → mu_contract → obs (3-level hierarchy)",
        "estimator_method": "MCMC (NUTS) or ADVI",
        "install": "pip install pymc",
    }
