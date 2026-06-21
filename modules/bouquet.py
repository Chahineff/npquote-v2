"""P0.3 — Bouquet : agrégation multi-tranches avec part uniforme ou variable.

Permet de tarifer un programme complet (T1 + T2 + T3 …) avec :
- Part uniforme (signing line homogène) ou variable par tranche
- Simulation Monte Carlo agrégée respectant la corrélation entre tranches
  (1 sinistre traversant T1 → T2 → T3 = 1 perte agrégée, pas 3 indépendantes)
- Calcul du capital diversifié au niveau bouquet
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .aad_aal import TrancheLayer, apply_layer_per_claim, apply_aad_aal_annual


@dataclass
class TrancheCotation:
    """Une tranche du bouquet avec sa part et sa prime commerciale."""
    layer: TrancheLayer
    part_reass: float            # 0..1, % cédé au réassureur (signing line)
    prime_commerciale_100pct: float  # à 100% du marché


@dataclass
class BouquetCotation:
    """Cotation d'un bouquet : ensemble de tranches partageant la même
    expérience sinistre (même cédante, même portefeuille).
    """
    nom: str
    tranches: list[TrancheCotation]

    def parts(self) -> list[float]:
        return [t.part_reass for t in self.tranches]

    def primes_100pct(self) -> list[float]:
        return [t.prime_commerciale_100pct for t in self.tranches]

    def primes_part(self) -> list[float]:
        return [t.prime_commerciale_100pct * t.part_reass for t in self.tranches]

    def total_prime_part(self) -> float:
        return sum(self.primes_part())

    def uniformiser_part(self, part: float) -> None:
        """Applique le même % de cession à toutes les tranches."""
        for t in self.tranches:
            t.part_reass = part


def simulate_bouquet(
    rng: np.random.Generator,
    bouquet: BouquetCotation,
    n_years: int,
    n_distribution,    # callable(rng, n_years) -> array of int
    x_distribution,    # callable(rng, n_claims) -> array of severity
) -> dict:
    """Simulation Monte Carlo avec corrélation parfaite intra-année.

    Pour chaque année :
      1. Tire N sinistres avec la loi de fréquence
      2. Tire N coûts avec la loi de sévérité
      3. Applique CHAQUE tranche aux MÊMES sinistres (corrélation = 1)
      4. Somme par tranche puis applique AAD/AAL et part réass

    Retourne :
      - charge_par_tranche : (n_tranches, n_years) charge cédée au réass
      - charge_bouquet     : (n_years,) somme des charges cédées
      - stats agrégées
    """
    K = len(bouquet.tranches)
    charge_par_tranche = np.zeros((K, n_years))

    nb_per_year = n_distribution(rng, n_years)
    for y in range(n_years):
        n_y = int(nb_per_year[y])
        if n_y == 0:
            continue
        sev = x_distribution(rng, n_y)
        for k, tc in enumerate(bouquet.tranches):
            layer = tc.layer
            per_claim = apply_layer_per_claim(sev, layer)
            annual = per_claim.sum()
            ceded_100pct = apply_aad_aal_annual(np.array([annual]), layer)[0]
            charge_par_tranche[k, y] = ceded_100pct * tc.part_reass

    charge_bouquet = charge_par_tranche.sum(axis=0)

    stats_par_tranche = []
    for k, tc in enumerate(bouquet.tranches):
        ch = charge_par_tranche[k]
        mean = ch.mean()
        var995 = float(np.quantile(ch, 0.995))
        q99 = np.quantile(ch, 0.99)
        tvar99 = float(ch[ch >= q99].mean()) if (ch >= q99).any() else q99
        stats_par_tranche.append({
            "nom": tc.layer.name,
            "part": tc.part_reass,
            "prime_100pct": tc.prime_commerciale_100pct,
            "prime_part": tc.prime_commerciale_100pct * tc.part_reass,
            "charge_attendue_part": float(mean),
            "ecart_type_part": float(ch.std(ddof=0)),
            "var_99_5": var995,
            "tvar_99": tvar99,
            "capital_s2_standalone": float(max(var995 - mean, 0)),
        })

    mean_b = float(charge_bouquet.mean())
    std_b = float(charge_bouquet.std(ddof=0))
    var995_b = float(np.quantile(charge_bouquet, 0.995))
    q99_b = np.quantile(charge_bouquet, 0.99)
    tvar99_b = (float(charge_bouquet[charge_bouquet >= q99_b].mean())
                if (charge_bouquet >= q99_b).any() else float(q99_b))
    cap_b = max(var995_b - mean_b, 0)
    cap_standalone_sum = sum(s["capital_s2_standalone"] for s in stats_par_tranche)
    diversification_benefit = (1 - cap_b / cap_standalone_sum
                               if cap_standalone_sum > 0 else 0)

    return {
        "charge_par_tranche": charge_par_tranche,
        "charge_bouquet": charge_bouquet,
        "stats_par_tranche": stats_par_tranche,
        "stats_bouquet": {
            "nom": bouquet.nom,
            "charge_attendue": mean_b,
            "ecart_type": std_b,
            "var_99_5": var995_b,
            "tvar_99": tvar99_b,
            "capital_s2_diversifie": cap_b,
            "capital_s2_somme_standalone": cap_standalone_sum,
            "benefice_diversification_pct": diversification_benefit,
            "prime_totale_part": bouquet.total_prime_part(),
            "loss_ratio": mean_b / bouquet.total_prime_part()
                         if bouquet.total_prime_part() > 0 else 0,
        },
        "quantiles_bouquet": {q: float(np.quantile(charge_bouquet, q))
                              for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]},
    }


def signing_optimization(
    rng: np.random.Generator,
    bouquet: BouquetCotation,
    n_years: int,
    n_distribution,
    x_distribution,
    target_roe: float = 0.12,
    max_part: float = 0.30,
    capital_to_prime_ratio: float = 0.30,
) -> dict:
    """Trouve la signing line optimale par tranche maximisant le RoE bouquet.

    Approche : grid search sur les % de cession par tranche.
    Pour des bouquets > 3 tranches, préférer une optim convexe (cvxpy).
    """
    K = len(bouquet.tranches)
    if K > 4:
        return {"warning": "Grid search limité à 4 tranches, utiliser optim convexe"}

    grid = np.linspace(0.05, max_part, 6)
    best = None
    for combo in _product(grid, K):
        for i, p in enumerate(combo):
            bouquet.tranches[i].part_reass = p
        res = simulate_bouquet(rng, bouquet, n_years, n_distribution, x_distribution)
        s = res["stats_bouquet"]
        prime = s["prime_totale_part"]
        if prime <= 0:
            continue
        result_tech = prime - s["charge_attendue"]
        capital = max(s["capital_s2_diversifie"],
                      capital_to_prime_ratio * prime)
        roe = result_tech / capital if capital > 0 else 0
        if best is None or roe > best["roe"]:
            best = {"combo": list(combo), "roe": roe,
                    "prime": prime, "loss_ratio": s["loss_ratio"]}

    if best:
        for i, p in enumerate(best["combo"]):
            bouquet.tranches[i].part_reass = p
    return best


def _product(grid, K):
    """Itère sur le produit cartésien d'une grille à K dimensions."""
    if K == 1:
        for x in grid:
            yield (x,)
    else:
        for x in grid:
            for rest in _product(grid, K - 1):
                yield (x,) + rest
