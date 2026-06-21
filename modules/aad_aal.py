"""P0.4 — AAD / AAL pour Burning Cost et Monte Carlo.

Conditions annuelles agrégées :
- AAD (Annual Aggregate Deductible) : franchise annuelle agrégée
- AAL (Annual Aggregate Limit)      : plafond annuel agrégé
- Reinstatements payants/gratuits (Nrec, %Txrec)

L'outil v1 ne les applique qu'en Simulation ; ici on les applique aussi
en Burning Cost (sommation année par année de la charge tranche).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TrancheLayer:
    """Structure d'une tranche XL avec toutes ses conditions annuelles."""
    name: str
    priority: float          # rétention (XS)
    limit: float             # garantie au-dessus de la priorité
    aad: float = 0.0         # franchise annuelle agrégée
    aal: float = float('inf')  # plafond annuel agrégé (par défaut illimité)
    n_reinstatements: int = 999
    txrec: list[float] = None  # % reconstitution payante (0..1)

    def __post_init__(self):
        if self.txrec is None:
            self.txrec = [1.0] * self.n_reinstatements
        else:
            self.txrec = [float(x) for x in self.txrec]
        if self.aal == float('inf') and self.n_reinstatements < 999:
            self.aal = self.limit * (1 + self.n_reinstatements)


def apply_layer_per_claim(claims: np.ndarray, layer: TrancheLayer) -> np.ndarray:
    """Applique priorité + plafond par sinistre (avant agrégation annuelle).

    Pour chaque sinistre x : charge tranche = min(max(x - priorité, 0), limit)
    """
    return np.clip(claims - layer.priority, 0, layer.limit)


def apply_aad_aal_annual(
    annual_losses_per_layer: np.ndarray, layer: TrancheLayer
) -> np.ndarray:
    """Applique AAD puis AAL à la charge annuelle agrégée d'une tranche.

    annual_losses_per_layer : tableau 1D, charge tranche par année
                              (déjà passée par apply_layer_per_claim et sommée)
    Retourne la charge cédée au réassureur après AAD/AAL.
    """
    after_aad = np.maximum(annual_losses_per_layer - layer.aad, 0)
    after_aal = np.minimum(after_aad, layer.aal)
    return after_aal


def burning_cost_with_aad_aal(
    claims_by_year: dict[int, np.ndarray],
    layer: TrancheLayer,
    epi_by_year: dict[int, float],
    ibnr_factor: dict[int, float] | None = None,
    ignore_years: set[int] | None = None,
) -> dict:
    """Burning Cost CORRIGÉ AAD/AAL : version qui agrège par année.

    Différence vs v1 : v1 somme année par année MAIS n'applique pas AAD/AAL.
    Ici, on rentre dans l'algorithme correct :
      1. Pour chaque sinistre x : layer charge = clip(x - priorité, 0, limit)
      2. Somme par année → annual_layer_loss
      3. Appliquer AAD puis AAL → ceded loss
      4. BC année = ceded_loss / EPI année
      5. Tarif = moyenne des BC année × IBNR

    Retourne dict avec : bc_annual, bc_mean, bc_std, garantie_consommee,
                        nb_reinstatements_used_avg
    """
    ibnr_factor = ibnr_factor or {}
    ignore_years = ignore_years or set()

    years_used, bc_values, gar_conso, rec_used = [], [], [], []
    for year, claims in claims_by_year.items():
        if year in ignore_years:
            continue
        epi = epi_by_year.get(year, 0)
        if epi <= 0:
            continue
        per_claim = apply_layer_per_claim(np.asarray(claims), layer)
        annual = per_claim.sum()
        ceded = apply_aad_aal_annual(np.array([annual]), layer)[0]
        ibnr = ibnr_factor.get(year, 1.0)
        bc = (ceded * ibnr) / epi
        years_used.append(year)
        bc_values.append(bc)
        gar_conso.append(ceded / layer.limit if layer.limit > 0 else 0)
        rec_used.append(min(ceded / layer.limit, 1 + layer.n_reinstatements)
                        if layer.limit > 0 else 0)

    bc_arr = np.array(bc_values)
    return {
        "years": years_used,
        "bc_annual": bc_arr,
        "bc_mean": float(bc_arr.mean()) if len(bc_arr) else 0.0,
        "bc_std": float(bc_arr.std(ddof=0)) if len(bc_arr) else 0.0,
        "garantie_consommee_avg": float(np.mean(gar_conso)) if gar_conso else 0.0,
        "nb_reinstatements_used_avg": float(np.mean(rec_used)) if rec_used else 0.0,
        "stat_duration": len(years_used),
    }


def reinstatement_reduction(layer: TrancheLayer, rec_used_avg: float) -> float:
    """Facteur de réduction du tarif dû aux primes de reconstitution payantes.

    Formule v1 NPquote :
      r = Nr / (Nr + N0)
      où Nr = somme des % de rec payantes consommées,
         N0 = nb d'années (= nb de "primes de base" perçues)
    Le tarif est multiplié par (1 - r).
    """
    if not layer.txrec or rec_used_avg <= 0:
        return 0.0
    nr = 0.0
    remaining = rec_used_avg
    for tx in layer.txrec:
        used = min(remaining, 1.0)
        nr += tx * used
        remaining -= used
        if remaining <= 0:
            break
    return nr / (nr + 1.0) if nr + 1.0 > 0 else 0.0


def monte_carlo_with_aad_aal(
    rng: np.random.Generator,
    n_years: int,
    n_distribution,     # callable(rng, n_years) -> array of int (claim count)
    x_distribution,     # callable(rng, n_claims) -> array of severity
    layer: TrancheLayer,
) -> dict:
    """Monte Carlo XL avec AAD/AAL appliqués au niveau annuel.

    Retourne distribution complète de la charge cédée + tarif + stats.
    """
    ceded_per_year = np.zeros(n_years)
    n_touches = 0
    n_full_consumed = 0

    nb_per_year = n_distribution(rng, n_years)
    for i in range(n_years):
        n_i = int(nb_per_year[i])
        if n_i == 0:
            continue
        sev = x_distribution(rng, n_i)
        layer_losses = apply_layer_per_claim(sev, layer)
        annual = layer_losses.sum()
        ceded = apply_aad_aal_annual(np.array([annual]), layer)[0]
        ceded_per_year[i] = ceded
        if ceded > 0:
            n_touches += 1
        if ceded >= layer.aal - 1e-6 and layer.aal < float('inf'):
            n_full_consumed += 1

    return {
        "ceded_per_year": ceded_per_year,
        "mean": float(ceded_per_year.mean()),
        "std": float(ceded_per_year.std(ddof=0)),
        "p_touch": n_touches / n_years,
        "p_full": n_full_consumed / n_years,
        "var_99_5": float(np.quantile(ceded_per_year, 0.995)),
        "tvar_99": float(ceded_per_year[ceded_per_year >=
                         np.quantile(ceded_per_year, 0.99)].mean())
                    if (ceded_per_year >= np.quantile(ceded_per_year, 0.99)).any()
                    else 0.0,
        "quantiles": {q: float(np.quantile(ceded_per_year, q))
                      for q in [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]},
    }
