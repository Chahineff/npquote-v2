"""P2.2 — RAROC complet + Solvency II SCR + EVA.

Décomposition complète du capital économique selon la formule standard
Solvency II :

  SCR_total² = SCR_NL_Prem_Res² + SCR_NL_Cat² + 2 × ρ × SCR_NL_Prem_Res × SCR_NL_Cat

Référence : Delegated Regulation (EU) 2015/35, articles 114-135.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# Volatilité standard Solvency II par S2 LoB Non-Life
# Source : Annexe II du Règlement Délégué (UE) 2015/35
SII_LOB_VOLATILITY = {
    "motor_liability":           {"sigma_prem": 0.10, "sigma_res": 0.09},
    "motor_other":               {"sigma_prem": 0.08, "sigma_res": 0.08},
    "marine_aviation_transport": {"sigma_prem": 0.15, "sigma_res": 0.11},
    "fire_property":             {"sigma_prem": 0.08, "sigma_res": 0.10},
    "general_liability":         {"sigma_prem": 0.14, "sigma_res": 0.11},
    "credit_suretyship":         {"sigma_prem": 0.12, "sigma_res": 0.19},
    "legal_expense":             {"sigma_prem": 0.07, "sigma_res": 0.12},
    "assistance":                {"sigma_prem": 0.09, "sigma_res": 0.20},
    "miscellaneous":             {"sigma_prem": 0.13, "sigma_res": 0.20},
    "np_property":               {"sigma_prem": 0.17, "sigma_res": 0.20},
    "np_casualty":               {"sigma_prem": 0.17, "sigma_res": 0.20},
    "np_marine_aviation":        {"sigma_prem": 0.17, "sigma_res": 0.20},
}

# Matrice de corrélation Solvency II entre LoB (annexe IV)
SII_LOB_CORRELATION = {
    ("motor_liability", "fire_property"): 0.25,
    ("motor_liability", "general_liability"): 0.50,
    ("fire_property", "marine_aviation_transport"): 0.25,
    ("fire_property", "np_property"): 0.25,
    ("general_liability", "np_casualty"): 0.25,
}


@dataclass
class SCRComponents:
    """Décomposition complète du SCR Solvency II Non-Life."""
    scr_nl_premium: float        # SCR sur prime de l'année
    scr_nl_reserve: float        # SCR sur provisions techniques
    scr_nl_prem_res: float       # combiné prem + res avec corrélation 0.5
    scr_nl_cat: float            # SCR catastrophe
    scr_nl_total: float          # combiné avec rho prem_res / cat = 0.25
    scr_market: float = 0.0      # risque marché (placements actifs)
    scr_default: float = 0.0     # risque défaut contrepartie
    bscr: float = 0.0            # Basic SCR
    operational_risk: float = 0.0
    scr_total: float = 0.0

    def to_dict(self):
        return {
            "scr_nl_premium": self.scr_nl_premium,
            "scr_nl_reserve": self.scr_nl_reserve,
            "scr_nl_prem_res": self.scr_nl_prem_res,
            "scr_nl_cat": self.scr_nl_cat,
            "scr_nl_total": self.scr_nl_total,
            "scr_market": self.scr_market,
            "scr_default": self.scr_default,
            "bscr": self.bscr,
            "operational_risk": self.operational_risk,
            "scr_total": self.scr_total,
        }


def compute_scr_premium_reserve(
    premiums_by_lob: dict[str, float],
    reserves_by_lob: dict[str, float],
    lob_mapping: dict[str, str] = None,
) -> tuple[float, float, float]:
    """SCR_NL_Prem_Res selon Solvency II Standard Formula.

    Formule : V × 3 × σ_combined  (approximation USP avec quantile 99.5%)
    """
    lob_mapping = lob_mapping or {}
    V_p = sum(premiums_by_lob.values())
    V_r = sum(reserves_by_lob.values())
    V = V_p + V_r

    sigma_p_weighted = 0
    sigma_r_weighted = 0
    for lob_name, prem in premiums_by_lob.items():
        s2_lob = lob_mapping.get(lob_name, lob_name)
        sigmas = SII_LOB_VOLATILITY.get(s2_lob, {"sigma_prem": 0.10, "sigma_res": 0.10})
        sigma_p_weighted += prem * sigmas["sigma_prem"]
    for lob_name, res in reserves_by_lob.items():
        s2_lob = lob_mapping.get(lob_name, lob_name)
        sigmas = SII_LOB_VOLATILITY.get(s2_lob, {"sigma_prem": 0.10, "sigma_res": 0.10})
        sigma_r_weighted += res * sigmas["sigma_res"]

    sigma_p = sigma_p_weighted / V_p if V_p > 0 else 0
    sigma_r = sigma_r_weighted / V_r if V_r > 0 else 0

    sigma_pr2 = ((sigma_p * V_p) ** 2 + 2 * 0.5 * sigma_p * V_p * sigma_r * V_r
                  + (sigma_r * V_r) ** 2)
    sigma_pr = np.sqrt(sigma_pr2) / V if V > 0 else 0

    scr_nl_prem_res = 3 * sigma_pr * V
    scr_nl_prem = 3 * sigma_p * V_p
    scr_nl_res = 3 * sigma_r * V_r

    return scr_nl_prem, scr_nl_res, scr_nl_prem_res


def compute_scr_cat(
    pml_by_peril: dict[str, float],
    correlation_matrix: dict | None = None,
) -> float:
    """SCR_NL_Cat = √(Σᵢⱼ ρ_ij × PML_i × PML_j).

    pml_by_peril : {"natcat_eq": 50e6, "natcat_storm": 30e6, "manmade_fire": 20e6,...}
    """
    perils = list(pml_by_peril.keys())
    n = len(perils)
    pml = np.array([pml_by_peril[p] for p in perils])

    rho = np.eye(n)
    if correlation_matrix:
        for (i, j), val in correlation_matrix.items():
            if i in perils and j in perils:
                ii, jj = perils.index(i), perils.index(j)
                rho[ii, jj] = val
                rho[jj, ii] = val
    else:
        rho = np.eye(n) + 0.25 * (np.ones((n, n)) - np.eye(n))

    scr2 = float(pml @ rho @ pml)
    return np.sqrt(scr2)


def compute_full_scr(
    premiums_by_lob: dict[str, float],
    reserves_by_lob: dict[str, float],
    pml_by_peril: dict[str, float],
    *,
    scr_market: float = 0.0,
    scr_default: float = 0.0,
    op_risk_pct_premium: float = 0.03,
    lob_mapping: dict[str, str] = None,
) -> SCRComponents:
    """Calcule la décomposition complète du SCR.

    Formule officielle Solvency II :
      BSCR² = Σᵢⱼ Corr_ij × SCR_i × SCR_j
      SCR_total = BSCR + Op_Risk (additive, pas euclidien)
    """
    scr_prem, scr_res, scr_pr = compute_scr_premium_reserve(
        premiums_by_lob, reserves_by_lob, lob_mapping)
    scr_cat = compute_scr_cat(pml_by_peril) if pml_by_peril else 0

    # SCR NL total : combine prem_res et cat avec rho = 0.25
    scr_nl_total = np.sqrt(scr_pr ** 2 + 2 * 0.25 * scr_pr * scr_cat + scr_cat ** 2)

    # BSCR combine NL, Market, Default avec matrice de corrélation Solvency II
    scrs = np.array([scr_nl_total, scr_market, scr_default])
    rho = np.array([
        [1.00, 0.25, 0.50],
        [0.25, 1.00, 0.25],
        [0.50, 0.25, 1.00],
    ])
    bscr = float(np.sqrt(scrs @ rho @ scrs))

    op_risk = op_risk_pct_premium * sum(premiums_by_lob.values())
    scr_total = bscr + op_risk

    return SCRComponents(
        scr_nl_premium=scr_prem, scr_nl_reserve=scr_res, scr_nl_prem_res=scr_pr,
        scr_nl_cat=scr_cat, scr_nl_total=scr_nl_total,
        scr_market=scr_market, scr_default=scr_default,
        bscr=bscr, operational_risk=op_risk, scr_total=scr_total,
    )


@dataclass
class RarocResult:
    """Résultat RAROC complet pour une cotation."""
    nom: str
    prime_commerciale: float
    sinistres_attendus: float
    expenses_total: float
    risk_capital: float           # capital économique
    cost_of_capital_pct: float    # hurdle rate (e.g. 10%)
    cost_of_capital: float
    raw_net_income: float         # avant cost of capital
    economic_profit: float        # = RT - CoC
    raroc: float                  # (RT - expected loss only) / capital
    rorac: float                  # RT / capital
    eva: float                    # Economic Value Added = RT - CoC
    sharpe_ratio: float = 0.0
    irr: float = 0.0

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


def compute_raroc(
    nom: str,
    *,
    prime_commerciale: float,
    sinistres_attendus: float,
    sinistres_volatility: float,    # std des sinistres
    expenses_total: float,
    var_99_5: float,
    cost_of_capital_pct: float = 0.10,
    discount_rate: float = 0.03,
    duration_years: float = 1.0,
) -> RarocResult:
    """Calcule RAROC, RORAC, EVA et Sharpe ratio.

    RAROC = (Revenu - Pertes attendues) / Capital économique
    RORAC = Résultat technique / Capital économique
    EVA   = Résultat technique - Capital × hurdle_rate
    Sharpe (souscription) = (RT - rf × Capital) / σ_sinistres
    """
    risk_capital = max(var_99_5 - sinistres_attendus, 0.20 * prime_commerciale)
    coc = risk_capital * cost_of_capital_pct

    raw_net = prime_commerciale - sinistres_attendus - expenses_total
    economic_profit = raw_net - coc

    raroc = (prime_commerciale - sinistres_attendus - expenses_total) / risk_capital \
            if risk_capital > 0 else 0
    rorac = raw_net / risk_capital if risk_capital > 0 else 0
    eva = economic_profit

    sharpe = ((raw_net - discount_rate * risk_capital) / sinistres_volatility
              if sinistres_volatility > 0 else 0)

    # IRR approximée pour cotation 1 an : (RT / Capital) - hurdle
    irr_approx = rorac - cost_of_capital_pct

    return RarocResult(
        nom=nom, prime_commerciale=prime_commerciale,
        sinistres_attendus=sinistres_attendus, expenses_total=expenses_total,
        risk_capital=risk_capital, cost_of_capital_pct=cost_of_capital_pct,
        cost_of_capital=coc, raw_net_income=raw_net,
        economic_profit=economic_profit, raroc=raroc, rorac=rorac, eva=eva,
        sharpe_ratio=sharpe, irr=irr_approx,
    )


def hurdle_rate_premium(
    raroc_result: RarocResult,
    target_hurdle: float = 0.15,
) -> dict:
    """Calcule la prime additionnelle pour atteindre un hurdle rate donné.

    Si RAROC < hurdle, on cherche l'augmentation de prime nécessaire.
    Δprime = Capital × (hurdle - RAROC actuel)
    """
    gap = max(target_hurdle - raroc_result.raroc, 0)
    delta_premium = raroc_result.risk_capital * gap
    new_premium = raroc_result.prime_commerciale + delta_premium
    pct_increase = delta_premium / raroc_result.prime_commerciale \
                    if raroc_result.prime_commerciale > 0 else 0

    return {
        "current_raroc": raroc_result.raroc,
        "target_hurdle": target_hurdle,
        "gap_bps": gap * 10000,
        "delta_premium_required": delta_premium,
        "new_premium_required": new_premium,
        "pct_increase_required": pct_increase,
        "verdict": "✓ OK" if gap == 0 else "✗ Insufficient — request rate increase",
    }
