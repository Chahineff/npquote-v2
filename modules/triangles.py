"""P1.6 — Triangles de liquidation : Chain Ladder / Mack / BF / Cape Cod.

Estime les IBNR à partir de triangles cumulés (paiements ou charges).
Output IBNR par année d'origine → alimente le coeff_IBNR de NPquote.

Référence : Mack (1993) "Distribution-free calculation of standard error of
chain ladder reserve estimates", ASTIN Bulletin.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TriangleResult:
    method: str
    triangle_completed: pd.DataFrame  # triangle complété (carré)
    ultimates: pd.Series              # ultimes par année d'origine
    reserves: pd.Series               # IBNR par année d'origine
    development_factors: pd.Series    # f_j par delay
    ibnr_factors: pd.Series           # ratio ultime / observé (= coeff IBNR)
    mse: float | None = None          # mean square error (Mack uniquement)
    se_by_origin: pd.Series | None = None
    cv_by_origin: pd.Series | None = None


def chain_ladder(triangle: pd.DataFrame) -> TriangleResult:
    """Chain Ladder classique sur triangle cumulé.

    triangle : DataFrame index=année d'origine, colonnes=delay (0,1,2,...).
               Valeurs cumulées en haut, NaN dans le coin sud-est.
    """
    tri = triangle.copy()
    n_origins, n_dev = tri.shape

    # Facteurs de développement
    f = []
    for j in range(n_dev - 1):
        num = tri.iloc[:, j + 1].dropna().sum()
        den = tri.iloc[:tri.iloc[:, j + 1].dropna().shape[0], j].sum()
        f.append(num / den if den > 0 else 1.0)
    f = pd.Series(f, index=tri.columns[1:], name="f_j")

    # Complétion du triangle
    completed = tri.copy()
    for i in range(n_origins):
        for j in range(n_dev - 1):
            if pd.isna(completed.iloc[i, j + 1]) and not pd.isna(completed.iloc[i, j]):
                completed.iloc[i, j + 1] = completed.iloc[i, j] * f.iloc[j]

    ultimates = completed.iloc[:, -1]
    last_observed = tri.apply(lambda row: row.dropna().iloc[-1] if row.dropna().any()
                              else np.nan, axis=1)
    reserves = ultimates - last_observed
    ibnr_factors = ultimates / last_observed.where(last_observed != 0, np.nan)

    return TriangleResult(
        method="Chain Ladder",
        triangle_completed=completed,
        ultimates=ultimates,
        reserves=reserves,
        development_factors=f,
        ibnr_factors=ibnr_factors,
    )


def mack_method(triangle: pd.DataFrame) -> TriangleResult:
    """Méthode de Mack : Chain Ladder + intervalles de confiance distribution-free.

    Calcule l'erreur quadratique moyenne (MSE) des réserves selon la formule de
    Mack (1993).
    """
    cl = chain_ladder(triangle)
    tri = triangle.copy()
    n_origins, n_dev = tri.shape
    f = cl.development_factors.values

    # Estimation des sigma_j² (Mack 1993)
    sigma2 = []
    for j in range(n_dev - 1):
        col_j = tri.iloc[:, j].dropna()
        col_jp1 = tri.iloc[:, j + 1].dropna()
        # Indices communs
        idx_common = col_j.index.intersection(col_jp1.index)
        if len(idx_common) < 2:
            sigma2.append(sigma2[-1] if sigma2 else 0)
            continue
        residuals = []
        for i in idx_common:
            cij = tri.loc[i].iloc[j]
            cij1 = tri.loc[i].iloc[j + 1]
            if cij > 0:
                residuals.append(cij * (cij1 / cij - f[j]) ** 2)
        s2 = sum(residuals) / max(len(idx_common) - 1, 1)
        sigma2.append(s2)

    # MSE par année d'origine (formule Mack)
    completed = cl.triangle_completed
    mse_by_origin = pd.Series(0.0, index=tri.index)
    for i in range(n_origins):
        ult = completed.iloc[i, -1]
        # Indice du dernier delay observé pour cette année
        last_j = tri.iloc[i].dropna().index[-1] if tri.iloc[i].dropna().any() else None
        if last_j is None:
            continue
        last_j_pos = tri.columns.get_loc(last_j)
        mse_i = 0
        for j in range(last_j_pos, n_dev - 1):
            cij = completed.iloc[i, j]
            denom_sum = tri.iloc[:n_origins - j - 1, j].sum() if j < n_dev - 1 else 1
            term = sigma2[j] / (f[j] ** 2) * (1 / cij + 1 / max(denom_sum, 1))
            mse_i += term * ult ** 2
        mse_by_origin.iloc[i] = mse_i

    se = np.sqrt(mse_by_origin)
    cv = se / cl.ultimates.where(cl.ultimates != 0, np.nan)

    return TriangleResult(
        method="Mack",
        triangle_completed=cl.triangle_completed,
        ultimates=cl.ultimates,
        reserves=cl.reserves,
        development_factors=cl.development_factors,
        ibnr_factors=cl.ibnr_factors,
        mse=float(mse_by_origin.sum()),
        se_by_origin=se,
        cv_by_origin=cv,
    )


def bornhuetter_ferguson(
    triangle: pd.DataFrame,
    a_priori_ultimates: pd.Series,
) -> TriangleResult:
    """Bornhuetter-Ferguson : combine a priori expert et CL pour années récentes.

    a_priori_ultimates : ultimes attendus par année d'origine (ex : prime × LR).
    Particulièrement utile pour les années récentes peu développées.
    """
    cl = chain_ladder(triangle)
    n_origins, n_dev = triangle.shape

    # Cumulative Development Factor : CDF_i = produit des f à partir de la position i
    cdf = []
    cumprod = 1.0
    for j in range(n_dev - 2, -1, -1):
        cumprod *= cl.development_factors.iloc[j]
        cdf.insert(0, cumprod)
    cdf.append(1.0)
    cdf = pd.Series(cdf, index=triangle.columns, name="cdf")
    pct_reported = 1 / cdf

    ultimates_bf = pd.Series(0.0, index=triangle.index)
    last_observed = triangle.apply(
        lambda row: row.dropna().iloc[-1] if row.dropna().any() else np.nan, axis=1)
    pct_at_last = triangle.apply(
        lambda row: pct_reported.iloc[len(row.dropna()) - 1]
                    if row.dropna().any() else np.nan, axis=1)
    for i, ay in enumerate(triangle.index):
        if ay in a_priori_ultimates.index:
            a = a_priori_ultimates.loc[ay]
            p = pct_at_last.loc[ay] if not pd.isna(pct_at_last.loc[ay]) else 1
            ultimates_bf.iloc[i] = last_observed.iloc[i] + a * (1 - p)
        else:
            ultimates_bf.iloc[i] = cl.ultimates.iloc[i]

    reserves_bf = ultimates_bf - last_observed
    ibnr_factors = ultimates_bf / last_observed.where(last_observed != 0, np.nan)

    return TriangleResult(
        method="Bornhuetter-Ferguson",
        triangle_completed=cl.triangle_completed,
        ultimates=ultimates_bf,
        reserves=reserves_bf,
        development_factors=cl.development_factors,
        ibnr_factors=ibnr_factors,
    )


def cape_cod(
    triangle: pd.DataFrame,
    premium_by_year: pd.Series,
) -> TriangleResult:
    """Cape Cod : variante de BF où le LR a priori est estimé sur le triangle.

    LR_CapeCod = somme(observed) / somme(premium × pct_reported)
    """
    cl = chain_ladder(triangle)
    n_origins, n_dev = triangle.shape

    cdf = []
    cumprod = 1.0
    for j in range(n_dev - 2, -1, -1):
        cumprod *= cl.development_factors.iloc[j]
        cdf.insert(0, cumprod)
    cdf.append(1.0)
    cdf = pd.Series(cdf, index=triangle.columns)
    pct_reported = 1 / cdf

    last_observed = triangle.apply(
        lambda row: row.dropna().iloc[-1] if row.dropna().any() else np.nan, axis=1)
    pct_at_last = triangle.apply(
        lambda row: pct_reported.iloc[len(row.dropna()) - 1]
                    if row.dropna().any() else np.nan, axis=1)

    num = last_observed.sum()
    den = (premium_by_year.reindex(triangle.index) * pct_at_last).sum()
    lr_cape_cod = num / den if den > 0 else 0

    a_priori = premium_by_year.reindex(triangle.index) * lr_cape_cod
    result = bornhuetter_ferguson(triangle, a_priori)
    result.method = f"Cape Cod (LR={lr_cape_cod:.1%})"
    return result


def ibnr_factors_for_npquote(
    triangle: pd.DataFrame,
    method: str = "chain_ladder",
    a_priori_ultimates: pd.Series | None = None,
    premium_by_year: pd.Series | None = None,
) -> pd.Series:
    """Retourne les facteurs IBNR par année à injecter dans NPquote.

    Coeff IBNR_année = Ultime / Observé courant
    À multiplier par la charge année dans le burning cost.
    """
    if method == "mack":
        res = mack_method(triangle)
    elif method == "bf" and a_priori_ultimates is not None:
        res = bornhuetter_ferguson(triangle, a_priori_ultimates)
    elif method == "cape_cod" and premium_by_year is not None:
        res = cape_cod(triangle, premium_by_year)
    else:
        res = chain_ladder(triangle)
    return res.ibnr_factors
