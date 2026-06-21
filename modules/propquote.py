"""P1.5 — PropQuote : tarification des traités proportionnels.

- Quote-Part (QS) avec sliding scale commission + profit commission
- Surplus avec courbes d'exposition et cession variable par police
- Compatible avec le compte technique du module compte_technique
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .compte_technique import CompteTechnique, Chargements


@dataclass
class SlidingScale:
    """Échelle de commission glissante (sliding scale).

    Commission = pivot + slope × (LR_pivot - LR_observed),
    bornée entre commission_min et commission_max.

    Exemple : pivot=30%, slope=1.0, LR_pivot=60%, min=25%, max=40%
      → LR=55% : commission = 30% + 1.0×(60%-55%) = 35%
      → LR=70% : commission = 30% + 1.0×(60%-70%) = 20% → clamp à 25%
    """
    commission_pivot: float = 0.30
    lr_pivot: float = 0.60
    slope: float = 1.0
    commission_min: float = 0.25
    commission_max: float = 0.40

    def commission(self, lr: float) -> float:
        c = self.commission_pivot + self.slope * (self.lr_pivot - lr)
        return float(np.clip(c, self.commission_min, self.commission_max))


@dataclass
class ProfitCommission:
    """Commission de profit sur le résultat technique positif de la cédante.

    PC = profit_share × max(0, prime_cedee - sinistres_cedes
                              - commission_cedante - frais_reass - reportable_loss)
    """
    profit_share: float = 0.20    # % du résultat positif rétrocédé à la cédante
    expense_allowance: float = 0.05  # frais réass dans la formule
    loss_carry_forward: bool = True


@dataclass
class QSCotation:
    """Cotation d'un traité Quote-Part."""
    nom: str
    cession_pct: float            # % cédé (par ex 25%)
    prime_brute_portfolio: float  # prime brute du portefeuille cédant
    lr_attendu: float             # loss ratio attendu (after indexation/asif)
    sliding_scale: SlidingScale | None = None
    profit_commission: ProfitCommission | None = None
    fixed_commission: float = 0.30  # si pas de sliding scale
    frais_reass_pct: float = 0.03   # frais gestion réassureur


def price_quote_part(qs: QSCotation, n_scenarios: int = 1000,
                     lr_volatility: float = 0.10,
                     rng: np.random.Generator | None = None) -> dict:
    """Tarifie un QS avec scénarios stochastiques sur le LR.

    Simule des LR autour du LR_attendu avec écart-type lr_volatility,
    calcule la commission par scénario, puis agrège.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    prime_cedee = qs.prime_brute_portfolio * qs.cession_pct
    lrs = rng.normal(qs.lr_attendu, lr_volatility, n_scenarios)
    lrs = np.clip(lrs, 0, 3.0)

    sinistres_cedes = lrs * prime_cedee

    if qs.sliding_scale:
        commissions_pct = np.array([qs.sliding_scale.commission(lr) for lr in lrs])
    else:
        commissions_pct = np.full(n_scenarios, qs.fixed_commission)
    commissions = commissions_pct * prime_cedee

    frais_reass = prime_cedee * qs.frais_reass_pct

    pc_amounts = np.zeros(n_scenarios)
    if qs.profit_commission:
        pc = qs.profit_commission
        base = prime_cedee - sinistres_cedes - commissions - prime_cedee * pc.expense_allowance
        pc_amounts = np.maximum(0, base) * pc.profit_share

    resultat_reass = prime_cedee - sinistres_cedes - commissions - frais_reass - pc_amounts

    return {
        "prime_cedee": prime_cedee,
        "lr_mean": float(lrs.mean()),
        "lr_std": float(lrs.std()),
        "commission_pct_mean": float(commissions_pct.mean()),
        "sinistres_attendus": float(sinistres_cedes.mean()),
        "commissions_attendues": float(commissions.mean()),
        "frais_reass": float(frais_reass),
        "profit_commission_attendue": float(pc_amounts.mean()),
        "resultat_attendu": float(resultat_reass.mean()),
        "resultat_std": float(resultat_reass.std()),
        "p_perte": float((resultat_reass < 0).mean()),
        "var_99_5": float(np.quantile(-resultat_reass, 0.995)),
        "tvar_99": float((-resultat_reass)[(-resultat_reass) >=
                          np.quantile(-resultat_reass, 0.99)].mean()),
        "combined_ratio_mean": float((sinistres_cedes + commissions
                                       + frais_reass + pc_amounts).mean()
                                     / prime_cedee),
    }


@dataclass
class SurplusCotation:
    """Cotation d'un traité Surplus.

    Surplus : pour chaque police, cession = (SI_police - retention_cedante) /
                                            SI_police, plafonné à n_lines × retention.
    """
    nom: str
    retention_cedante: float       # plein de conservation cédant (par police)
    n_lines: int                   # nombre de pleins du surplus
    portfolio_profile: pd.DataFrame  # colonnes : si_min, si_max, prime, sinistralite_lr
    courbe_expo_func: callable = None  # f(s) = % charge par % SI utilisé pour PML
    commission_pct: float = 0.30
    frais_reass_pct: float = 0.03


def _cession_par_bande(si_moyen: float, retention: float, n_lines: int) -> float:
    """Calcule la part cédée au surplus pour une police de SI donné."""
    if si_moyen <= retention:
        return 0.0
    surplus_capacity = retention * n_lines
    cedant_keeps = retention
    ceded_si = min(si_moyen - cedant_keeps, surplus_capacity)
    return ceded_si / si_moyen


def price_surplus(sp: SurplusCotation) -> dict:
    """Tarifie un Surplus à partir d'un profil portefeuille.

    Pour chaque bande d'engagement :
      1. Calcule la cession % moyenne
      2. Applique au prime → prime cédée
      3. Applique au LR → sinistres cédés
    """
    df = sp.portfolio_profile.copy()
    df['si_moyen'] = (df['si_min'] + df['si_max']) / 2
    df['cession_pct'] = df['si_moyen'].apply(
        lambda si: _cession_par_bande(si, sp.retention_cedante, sp.n_lines)
    )
    df['prime_cedee'] = df['prime'] * df['cession_pct']

    if 'sinistralite_lr' not in df.columns:
        df['sinistralite_lr'] = 0.6
    df['sinistres_cedes'] = df['prime_cedee'] * df['sinistralite_lr']

    total_prime_cedee = df['prime_cedee'].sum()
    total_sinistres = df['sinistres_cedes'].sum()
    commission = total_prime_cedee * sp.commission_pct
    frais = total_prime_cedee * sp.frais_reass_pct
    resultat = total_prime_cedee - total_sinistres - commission - frais

    return {
        "prime_cedee_totale": float(total_prime_cedee),
        "sinistres_cedes_attendus": float(total_sinistres),
        "commissions": float(commission),
        "frais_reass": float(frais),
        "resultat_attendu": float(resultat),
        "loss_ratio": float(total_sinistres / total_prime_cedee
                            if total_prime_cedee else 0),
        "combined_ratio": float((total_sinistres + commission + frais)
                                / total_prime_cedee if total_prime_cedee else 0),
        "detail_par_bande": df.to_dict(orient='records'),
    }


def qs_to_compte_technique(qs_result: dict, nom: str) -> CompteTechnique:
    """Convertit un résultat QS en CompteTechnique unifié."""
    prime = qs_result["prime_cedee"]
    return CompteTechnique(
        nom=nom,
        prime_pure=qs_result["sinistres_attendus"],
        prime_commerciale=prime,
        sinistres_attendus=qs_result["sinistres_attendus"],
        frais_acquisition=qs_result["commissions_attendues"],
        frais_gestion=qs_result["frais_reass"],
        autres_frais=qs_result["profit_commission_attendue"],
        capital_s2=qs_result["var_99_5"] - qs_result["sinistres_attendus"]
                    if qs_result["var_99_5"] > qs_result["sinistres_attendus"] else 0,
        tvar_99=qs_result["tvar_99"],
    )


def surplus_to_compte_technique(sp_result: dict, nom: str,
                                cv_lr: float = 0.20) -> CompteTechnique:
    """Convertit un résultat Surplus en CompteTechnique unifié.

    Hypothèse de capital : VaR99.5 ~ E[loss] × (1 + 2.576 × CV)
    """
    sinistres = sp_result["sinistres_cedes_attendus"]
    var = sinistres * (1 + 2.576 * cv_lr)
    return CompteTechnique(
        nom=nom,
        prime_pure=sinistres,
        prime_commerciale=sp_result["prime_cedee_totale"],
        sinistres_attendus=sinistres,
        frais_acquisition=sp_result["commissions"],
        frais_gestion=sp_result["frais_reass"],
        capital_s2=max(var - sinistres, 0),
        tvar_99=sinistres * (1 + 3.0 * cv_lr),
    )
