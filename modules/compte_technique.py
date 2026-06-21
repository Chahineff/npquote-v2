"""P0.2 — Compte Technique : LR / COR / Résultat / Capital S2 / RoE.

Module qui transforme un tarif technique en compte technique projeté
avec tous les KPI standards d'un underwriter réassurance.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Chargements:
    """Composantes du chargement appliqué à la prime pure."""
    courtage: float = 0.0          # % de la prime commerciale
    frais_gestion_reass: float = 0.0  # % réassureur
    frais_gestion_cedante: float = 0.0  # uniquement pour exposure
    securite_pct_sigma: float = 0.0  # k_sigma × écart-type ajouté
    autres_charges: float = 0.0    # marge de profit / divers
    profit_commission: float = 0.0  # % de PC sur résultat positif


@dataclass
class CompteTechnique:
    """Compte technique projeté d'une tranche ou d'un bouquet.

    Toutes les valeurs sont au niveau réassureur, à la part 100%
    (multiplier par la cession % pour avoir la vue SMGA).
    """
    nom: str
    prime_pure: float             # E[charge cédée]
    prime_commerciale: float      # prime brute encaissée
    sinistres_attendus: float     # = prime_pure (avant marges)
    frais_acquisition: float      # courtage
    frais_gestion: float          # frais gestion réassureur
    autres_frais: float = 0.0
    capital_s2: float = 0.0       # VaR 99.5% - E[charge]
    tvar_99: float = 0.0          # mesure cohérente du risque queue
    cout_capital_pct: float = 0.08  # 8% cible RoE défaut

    @property
    def loss_ratio(self):
        return self.sinistres_attendus / self.prime_commerciale \
               if self.prime_commerciale else 0

    @property
    def expense_ratio(self):
        total_fees = self.frais_acquisition + self.frais_gestion + self.autres_frais
        return total_fees / self.prime_commerciale if self.prime_commerciale else 0

    @property
    def combined_ratio(self):
        return self.loss_ratio + self.expense_ratio

    @property
    def resultat_technique(self):
        return (self.prime_commerciale - self.sinistres_attendus
                - self.frais_acquisition - self.frais_gestion - self.autres_frais)

    @property
    def cost_of_capital(self):
        return self.capital_s2 * self.cout_capital_pct

    @property
    def resultat_economique(self):
        return self.resultat_technique - self.cost_of_capital

    @property
    def roe_technique(self):
        return self.resultat_technique / self.capital_s2 if self.capital_s2 > 0 else 0

    @property
    def raroc(self):
        """Risk-Adjusted Return on Capital = RT / TVaR99 (mesure cohérente)."""
        return self.resultat_technique / self.tvar_99 if self.tvar_99 > 0 else 0

    def to_dict(self):
        d = asdict(self)
        d.update({
            "loss_ratio": self.loss_ratio,
            "expense_ratio": self.expense_ratio,
            "combined_ratio": self.combined_ratio,
            "resultat_technique": self.resultat_technique,
            "cost_of_capital": self.cost_of_capital,
            "resultat_economique": self.resultat_economique,
            "roe_technique": self.roe_technique,
            "raroc": self.raroc,
        })
        return d


def build_compte_from_simulation(
    nom: str,
    ceded_distribution: np.ndarray,
    prime_commerciale: float,
    chargements: Chargements,
    cout_capital_pct: float = 0.08,
) -> CompteTechnique:
    """Construit un CompteTechnique à partir d'une distribution Monte Carlo.

    ceded_distribution : array des charges cédées année par année (simulées).
    prime_commerciale  : prime brute négociée encaissée par le réassureur.
    """
    mean_loss = float(ceded_distribution.mean())
    var_99_5 = float(np.quantile(ceded_distribution, 0.995))
    q_99 = np.quantile(ceded_distribution, 0.99)
    mask = ceded_distribution >= q_99
    tvar_99 = float(ceded_distribution[mask].mean()) if mask.any() else q_99

    capital_s2 = max(var_99_5 - mean_loss, 0)
    frais_acq = prime_commerciale * chargements.courtage
    frais_ges = prime_commerciale * chargements.frais_gestion_reass
    autres = prime_commerciale * chargements.autres_charges

    return CompteTechnique(
        nom=nom,
        prime_pure=mean_loss,
        prime_commerciale=prime_commerciale,
        sinistres_attendus=mean_loss,
        frais_acquisition=frais_acq,
        frais_gestion=frais_ges,
        autres_frais=autres,
        capital_s2=capital_s2,
        tvar_99=tvar_99,
        cout_capital_pct=cout_capital_pct,
    )


def aggregate_comptes(
    comptes: list[CompteTechnique], shares: list[float], nom: str = "Bouquet"
) -> CompteTechnique:
    """Agrège plusieurs comptes techniques avec leurs parts respectives.

    Hypothèses : sinistres et capital additifs (pas de diversification).
    Pour un agrégat avec diversification, utiliser bouquet.aggregate_simulation().
    """
    assert len(comptes) == len(shares)

    primes_pures = sum(c.prime_pure * s for c, s in zip(comptes, shares))
    primes_comm = sum(c.prime_commerciale * s for c, s in zip(comptes, shares))
    sinistres = sum(c.sinistres_attendus * s for c, s in zip(comptes, shares))
    fa = sum(c.frais_acquisition * s for c, s in zip(comptes, shares))
    fg = sum(c.frais_gestion * s for c, s in zip(comptes, shares))
    autres = sum(c.autres_frais * s for c, s in zip(comptes, shares))
    cap = sum(c.capital_s2 * s for c, s in zip(comptes, shares))
    tvar = sum(c.tvar_99 * s for c, s in zip(comptes, shares))
    coc = comptes[0].cout_capital_pct if comptes else 0.08

    return CompteTechnique(
        nom=nom, prime_pure=primes_pures, prime_commerciale=primes_comm,
        sinistres_attendus=sinistres, frais_acquisition=fa, frais_gestion=fg,
        autres_frais=autres, capital_s2=cap, tvar_99=tvar,
        cout_capital_pct=coc,
    )


def technical_minimum_rate(
    chargements: Chargements,
    target_roe: float,
    expected_lr: float,
    capital_to_prime_ratio: float = 0.30,
) -> float:
    """Calcule le taux technique minimum (TMR) Group.

    Résout l'équation :
      RoE = (Prime × (1 - LR - Expenses)) / (Capital_ratio × Prime) = target
    →  taux_min = LR_attendu / (1 - Expenses - Capital_ratio × target_RoE)
    """
    expenses = (chargements.courtage + chargements.frais_gestion_reass
                + chargements.autres_charges)
    denom = 1 - expenses - capital_to_prime_ratio * target_roe
    if denom <= 0:
        return float('inf')
    return expected_lr / denom


def stress_test(
    compte: CompteTechnique,
    shocks: dict[str, float],
) -> dict[str, CompteTechnique]:
    """Calcule des scénarios stress-test sur le compte technique.

    shocks = {'+20% sinistres': 1.2, '+50% sinistres': 1.5, ...}
    """
    results = {}
    for label, multiplier in shocks.items():
        shocked = CompteTechnique(
            nom=f"{compte.nom} {label}",
            prime_pure=compte.prime_pure * multiplier,
            prime_commerciale=compte.prime_commerciale,
            sinistres_attendus=compte.sinistres_attendus * multiplier,
            frais_acquisition=compte.frais_acquisition,
            frais_gestion=compte.frais_gestion,
            autres_frais=compte.autres_frais,
            capital_s2=compte.capital_s2 * multiplier,
            tvar_99=compte.tvar_99 * multiplier,
            cout_capital_pct=compte.cout_capital_pct,
        )
        results[label] = shocked
    return results


def decision_engine(
    compte: CompteTechnique,
    tmr_taux: float,
    actual_taux: float,
    target_roe: float = 0.12,
    max_combined_ratio: float = 0.95,
) -> dict:
    """Moteur de décision : ACCEPT / REVISE / DECLINE.

    Compare le tarif négocié vs TMR et applique les seuils Group.
    """
    cor = compte.combined_ratio
    roe = compte.roe_technique
    ratio_vs_tmr = actual_taux / tmr_taux if tmr_taux > 0 else 0

    reasons = []
    if actual_taux < tmr_taux:
        reasons.append(f"Taux {actual_taux:.3%} < TMR {tmr_taux:.3%}")
    if cor > max_combined_ratio:
        reasons.append(f"COR {cor:.1%} > seuil {max_combined_ratio:.0%}")
    if roe < target_roe:
        reasons.append(f"RoE {roe:.1%} < cible {target_roe:.0%}")

    if not reasons:
        decision = "ACCEPT"
    elif ratio_vs_tmr >= 0.90 and cor < max_combined_ratio + 0.05:
        decision = "REVISE"
    else:
        decision = "DECLINE"

    return {
        "decision": decision,
        "reasons": reasons,
        "ratio_vs_tmr": ratio_vs_tmr,
        "combined_ratio": cor,
        "roe_technique": roe,
    }
