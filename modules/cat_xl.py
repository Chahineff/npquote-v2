"""P2.3 — Cat XL : ELT / AAL / OEP / AEP curves (format RMS/AIR).

Construit les sorties standardisées d'un cat modeling sur une couche XL :

- ELT (Event Loss Table)         : événement → (rate, mean_loss, std_loss)
- AAL (Average Annual Loss)      : pure premium attendue
- OEP (Occurrence Exceedance Prob): P(max loss > x) en 1 an
- AEP (Aggregate Exceedance Prob): P(sum losses > x) en 1 an

Compatible avec import direct depuis RMS RiskLink (.elt format) ou
AIR Touchstone export.

Référence : Grossi & Kunreuther (2005) "Catastrophe Modeling".
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import lognorm, poisson


@dataclass
class EventLossTable:
    """Event Loss Table standard cat modeling.

    Chaque événement i a :
      - rate_i  : fréquence annuelle Poisson
      - mean_i  : perte moyenne si l'événement survient
      - std_i   : écart-type (utilisé avec LogNormale pour la secondary uncertainty)
      - exposure_i : optionnel, exposition modélisée
    """
    events: pd.DataFrame   # colonnes : event_id, rate, mean_loss, std_loss, exposure

    @classmethod
    def from_rms(cls, csv_path: str) -> "EventLossTable":
        """Importe un ELT depuis un export RMS RiskLink CSV.

        Colonnes attendues : EventID, Rate, MeanLoss, StdDev, Exposure
        """
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "EventID": "event_id", "Rate": "rate",
            "MeanLoss": "mean_loss", "StdDev": "std_loss",
            "Exposure": "exposure",
        })
        return cls(df)

    @classmethod
    def from_air(cls, csv_path: str) -> "EventLossTable":
        """Importe un ELT depuis un export AIR Touchstone CSV."""
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "Event_ID": "event_id", "Annual_Rate": "rate",
            "Mean_Loss": "mean_loss", "Std_Loss": "std_loss",
            "Exposure_Value": "exposure",
        })
        return cls(df)

    @classmethod
    def synthetic(cls, n_events: int, mean_loss_range: tuple[float, float],
                   total_rate: float, seed: int = 42) -> "EventLossTable":
        """Génère un ELT synthétique pour démo/test."""
        rng = np.random.default_rng(seed)
        rates = rng.dirichlet(np.ones(n_events)) * total_rate
        means = rng.uniform(*mean_loss_range, size=n_events)
        stds = means * rng.uniform(0.3, 0.8, size=n_events)
        return cls(pd.DataFrame({
            "event_id": [f"EVT_{i:05d}" for i in range(n_events)],
            "rate": rates, "mean_loss": means, "std_loss": stds,
            "exposure": means * 10,
        }))


def apply_layer_to_elt(
    elt: EventLossTable, priority: float, limit: float,
    n_simulations: int = 100_000, seed: int = 1234,
) -> pd.DataFrame:
    """Applique une couche XL à un ELT et calcule les pertes layer par événement.

    Pour chaque événement, on suppose que la perte est LogNormale autour
    de mean_loss (with std_loss). On calcule la perte cédée à la couche
    via espérance conditionnelle ou Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    df = elt.events.copy()
    layer_means, layer_stds = [], []

    for _, row in df.iterrows():
        m, s = float(row['mean_loss']), float(row['std_loss'])
        if m <= 0 or s <= 0:
            layer_means.append(0.0)
            layer_stds.append(0.0)
            continue
        sigma2 = np.log(1 + (s / m) ** 2)
        mu = np.log(m) - sigma2 / 2
        sample = rng.lognormal(mu, np.sqrt(sigma2), size=n_simulations)
        layer_loss = np.clip(sample - priority, 0, limit)
        layer_means.append(float(layer_loss.mean()))
        layer_stds.append(float(layer_loss.std()))

    df['layer_mean'] = layer_means
    df['layer_std'] = layer_stds
    df['layer_aal_contribution'] = df['rate'] * df['layer_mean']
    return df


def compute_aal(elt_with_layer: pd.DataFrame) -> float:
    """Average Annual Loss = somme des rate × layer_mean."""
    return float(elt_with_layer['layer_aal_contribution'].sum())


def compute_oep_curve(
    elt_with_layer: pd.DataFrame,
    return_periods: list[float] = None,
) -> pd.DataFrame:
    """Occurrence Exceedance Probability curve.

    OEP(x) = P(max loss en 1 an > x) = 1 - exp(-Σᵢ rate_i × P(L_i > x))

    Avec la perte d'un événement modélisée LogNormale.
    """
    return_periods = return_periods or [5, 10, 25, 50, 100, 200, 250, 500, 1000]

    # Domaine de valeurs sur lequel évaluer OEP
    max_layer = float(elt_with_layer['layer_mean'].max() +
                       2 * elt_with_layer['layer_std'].max())
    if max_layer == 0:
        return pd.DataFrame()
    xs = np.linspace(1, max_layer * 3, 500)

    results = []
    for x in xs:
        # P(L_i > x | event i happens), perte modélisée LogN(layer_mean, layer_std)
        cumulative_rate = 0
        for _, row in elt_with_layer.iterrows():
            m = float(row['layer_mean'])
            s = float(row['layer_std'])
            if m <= 0 or s <= 0:
                continue
            sigma2 = np.log(1 + (s / m) ** 2)
            mu = np.log(m) - sigma2 / 2
            p = 1 - lognorm.cdf(x, s=np.sqrt(sigma2), scale=np.exp(mu))
            cumulative_rate += row['rate'] * p
        oep = 1 - np.exp(-cumulative_rate)
        results.append({"loss": x, "oep": oep,
                        "return_period": 1 / oep if oep > 0 else np.inf})

    df_curve = pd.DataFrame(results)

    # Extraire les return periods spécifiques par interpolation
    points = []
    for T in return_periods:
        target_p = 1 / T
        # Trouver loss tq OEP = target_p
        if df_curve['oep'].max() < target_p:
            points.append({"return_period": T, "loss": 0, "method": "below curve"})
            continue
        idx = (df_curve['oep'] - target_p).abs().idxmin()
        points.append({"return_period": T,
                        "loss": float(df_curve.loc[idx, 'loss']),
                        "oep_realized": float(df_curve.loc[idx, 'oep'])})

    return pd.DataFrame(points)


def compute_aep_curve(
    elt_with_layer: pd.DataFrame,
    return_periods: list[float] = None,
    n_years_simulation: int = 100_000,
    seed: int = 1234,
) -> pd.DataFrame:
    """Aggregate Exceedance Probability curve via Monte Carlo.

    AEP(x) = P(somme annuelle > x).
    Pour chaque année simulée :
      - Tire le nombre d'événements N ~ Poisson(Σ rates)
      - Tire N événements multinomial avec proba ∝ rate_i
      - Tire les pertes layer LogN
      - Somme
    """
    return_periods = return_periods or [5, 10, 25, 50, 100, 200, 250, 500, 1000]
    rng = np.random.default_rng(seed)

    df = elt_with_layer[elt_with_layer['layer_mean'] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    rates = df['rate'].values
    means = df['layer_mean'].values
    stds = df['layer_std'].values
    weights = rates / rates.sum()
    total_rate = rates.sum()

    aggregate_losses = np.zeros(n_years_simulation)
    for y in range(n_years_simulation):
        n = rng.poisson(total_rate)
        if n == 0:
            continue
        chosen = rng.choice(len(rates), size=n, p=weights)
        sigma2 = np.log(1 + (stds[chosen] / means[chosen]) ** 2)
        mu = np.log(means[chosen]) - sigma2 / 2
        losses = rng.lognormal(mu, np.sqrt(sigma2))
        aggregate_losses[y] = losses.sum()

    points = []
    for T in return_periods:
        if T <= n_years_simulation:
            q = 1 - 1 / T
            loss = float(np.quantile(aggregate_losses, q))
            points.append({"return_period": T, "loss": loss, "aep": 1 / T})

    return pd.DataFrame(points)


def cat_xl_summary(
    elt: EventLossTable, priority: float, limit: float,
    n_reinstatements: int = 1, premium_rate: float = 0.05,
) -> dict:
    """Synthèse complète d'une cotation Cat XL.

    Inputs :
      - elt : Event Loss Table
      - priority, limit : structure de la couche XL
      - n_reinstatements : nb de reconstitutions
      - premium_rate : taux ROL proposé (% de limit)
    """
    elt_layer = apply_layer_to_elt(elt, priority, limit)
    aal = compute_aal(elt_layer)
    oep = compute_oep_curve(elt_layer)
    aep = compute_aep_curve(elt_layer)

    rol = premium_rate
    premium = rol * limit
    loss_ratio_attendu = aal / premium if premium > 0 else 0

    # ROL minimum pour couvrir AAL + 50% marge cat
    rol_minimum = (aal * 1.5) / limit if limit > 0 else 0

    return {
        "aal": aal,
        "rol_proposed": rol,
        "rol_minimum_required": rol_minimum,
        "premium": premium,
        "loss_ratio_attendu": loss_ratio_attendu,
        "limit": limit,
        "priority": priority,
        "n_reinstatements": n_reinstatements,
        "oep_curve": oep,
        "aep_curve": aep,
        "elt_with_layer": elt_layer[['event_id', 'rate', 'mean_loss',
                                      'layer_mean', 'layer_aal_contribution']],
        "n_events_modeled": len(elt.events),
        "n_events_touching_layer": int((elt_layer['layer_mean'] > 0).sum()),
        "concentration_top10_pct": float(
            elt_layer['layer_aal_contribution'].nlargest(10).sum() / aal
            if aal > 0 else 0
        ),
    }
