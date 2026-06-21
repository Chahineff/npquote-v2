"""P0.1 — DataCheck : contrôles de cohérence des inputs courtiers.

Première ligne de défense avant tarification : détecte erreurs de saisie,
doublons, anomalies, fraudes potentielles dans les statements courtiers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
import pandas as pd


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    rows: list = field(default_factory=list)
    metric: float | None = None


@dataclass
class DataCheckReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, code, severity, message, rows=None, metric=None):
        self.findings.append(Finding(code, severity, message, rows or [], metric))

    @property
    def errors(self):
        return [f for f in self.findings
                if f.severity in (Severity.ERROR, Severity.CRITICAL)]

    @property
    def is_blocking(self):
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    def to_dataframe(self):
        return pd.DataFrame([{
            "code": f.code, "severity": f.severity.value,
            "message": f.message, "rows": str(f.rows[:10]),
            "metric": f.metric,
        } for f in self.findings])

    def summary(self):
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts


def check_claims(
    claims: pd.DataFrame,
    epi: pd.DataFrame,
    *,
    garantie_max: float | None = None,
    yoy_epi_threshold: float = 0.30,
    benford_threshold_pvalue: float = 0.01,
    expected_currency: str | None = None,
) -> DataCheckReport:
    """Contrôles complets sur la statistique sinistre + EPI.

    claims : colonnes attendues ['annee','nom','cout','devise'?,'date'?]
    epi    : colonnes attendues ['annee','epi','nb_polices'?,'devise'?]
    """
    report = DataCheckReport()

    _check_required_columns(claims, ['annee', 'cout'], 'claims', report)
    _check_required_columns(epi, ['annee', 'epi'], 'epi', report)
    if report.is_blocking:
        return report

    _check_numeric_validity(claims, 'cout', 'claims.cout', report)
    _check_numeric_validity(epi, 'epi', 'epi.epi', report)

    _check_year_gaps(epi['annee'], 'epi', report)
    _check_year_gaps(claims['annee'], 'claims', report, allow_gaps=True)

    _check_yoy_variation(epi, yoy_epi_threshold, report)

    if garantie_max is not None:
        excess = claims[claims['cout'] > garantie_max]
        if len(excess):
            report.add("CAP_001", Severity.WARNING,
                       f"{len(excess)} sinistres > garantie max ({garantie_max:,.0f})",
                       excess.index.tolist(), len(excess))

    _check_duplicates(claims, report)
    _check_currency(claims, epi, expected_currency, report)
    _check_benford(claims['cout'].dropna(), benford_threshold_pvalue, report)
    _check_frequency_coherence(claims, epi, report)
    _check_severity_outliers(claims['cout'].dropna(), report)
    _check_dev_year_zero_claims(claims, epi, report)

    return report


def _check_required_columns(df, required, name, report):
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add(f"COL_001", Severity.CRITICAL,
                   f"{name} : colonnes manquantes {missing}")


def _check_numeric_validity(df, col, name, report):
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        nb = s.apply(lambda x: not isinstance(x, (int, float))).sum()
        report.add("NUM_001", Severity.ERROR,
                   f"{name} : {nb} valeurs non-numériques", metric=nb)
        return
    nans = s.isna().sum()
    negs = (s < 0).sum()
    if nans:
        report.add("NUM_002", Severity.WARNING,
                   f"{name} : {nans} valeurs manquantes (NaN)", metric=nans)
    if negs:
        report.add("NUM_003", Severity.ERROR,
                   f"{name} : {negs} valeurs négatives", metric=negs)


def _check_year_gaps(years, name, report, allow_gaps=False):
    y = sorted(set(int(x) for x in years.dropna()))
    if not y:
        return
    expected = set(range(y[0], y[-1] + 1))
    missing = sorted(expected - set(y))
    if missing:
        sev = Severity.WARNING if allow_gaps else Severity.ERROR
        report.add("GAP_001", sev,
                   f"{name} : années manquantes {missing}",
                   metric=len(missing))


def _check_yoy_variation(epi, threshold, report):
    df = epi.sort_values('annee').reset_index(drop=True)
    df['yoy'] = df['epi'].pct_change()
    anomalies = df[df['yoy'].abs() > threshold]
    if len(anomalies):
        report.add("EPI_001", Severity.WARNING,
                   f"EPI : {len(anomalies)} variations YoY > {threshold:.0%}",
                   anomalies['annee'].tolist(),
                   metric=float(anomalies['yoy'].abs().max()))


def _check_duplicates(claims, report):
    cols_dup = [c for c in ['annee', 'nom', 'cout'] if c in claims.columns]
    if len(cols_dup) >= 2:
        dups = claims[claims.duplicated(subset=cols_dup, keep=False)]
        if len(dups):
            report.add("DUP_001", Severity.WARNING,
                       f"{len(dups)} doublons potentiels (clés: {cols_dup})",
                       dups.index.tolist()[:20], metric=len(dups))


def _check_currency(claims, epi, expected, report):
    for name, df in [('claims', claims), ('epi', epi)]:
        if 'devise' in df.columns:
            unique = df['devise'].dropna().unique()
            if len(unique) > 1:
                report.add("FX_001", Severity.ERROR,
                           f"{name} : devises mixtes {list(unique)}")
            elif expected and len(unique) == 1 and unique[0] != expected:
                report.add("FX_002", Severity.WARNING,
                           f"{name} : devise {unique[0]} != attendu {expected}")


def _check_benford(values, threshold_pvalue, report):
    """Test de Benford sur le premier chiffre des montants.

    Statistique : chi² sur 8 degrés de liberté.
    Une p-value très faible suggère que les montants ne suivent pas
    la loi de Benford → possible saisie manuelle/fabrication.
    """
    vals = values[values > 0]
    if len(vals) < 30:
        return
    first_digits = vals.apply(lambda x: int(str(x).lstrip('0').replace('.', '')[0])
                              if str(x).lstrip('0').replace('.', '') else 0)
    first_digits = first_digits[first_digits.between(1, 9)]
    if len(first_digits) < 30:
        return

    expected_p = {d: math.log10(1 + 1/d) for d in range(1, 10)}
    n = len(first_digits)
    chi2 = 0
    for d in range(1, 10):
        obs = (first_digits == d).sum()
        exp = expected_p[d] * n
        chi2 += (obs - exp) ** 2 / exp if exp > 0 else 0

    from scipy.stats import chi2 as chi2_dist
    pval = 1 - chi2_dist.cdf(chi2, df=8)
    if pval < threshold_pvalue:
        report.add("BEN_001", Severity.WARNING,
                   f"Test de Benford rejeté (chi²={chi2:.1f}, p={pval:.4f}). "
                   f"Premiers chiffres anormaux → vérifier saisie/fabrication",
                   metric=pval)


def _check_frequency_coherence(claims, epi, report):
    if 'annee' not in claims.columns:
        return
    freq = claims.groupby('annee').size().reset_index(name='nbsin')
    merged = freq.merge(epi[['annee', 'epi']], on='annee', how='inner')
    if len(merged) < 3:
        return
    merged['rate'] = merged['nbsin'] / merged['epi']
    mean_r = merged['rate'].mean()
    std_r = merged['rate'].std()
    if mean_r > 0 and std_r / mean_r > 1.0:
        report.add("FREQ_001", Severity.INFO,
                   f"Volatilité fréquence/EPI élevée : CV={std_r/mean_r:.2f}",
                   metric=std_r / mean_r)


def _check_severity_outliers(values, report):
    """Détection outliers par méthode IQR robuste."""
    vals = values[values > 0]
    if len(vals) < 10:
        return
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 5 * iqr
    outliers = vals[vals > upper]
    if len(outliers):
        ratio = outliers.max() / vals.median()
        report.add("OUT_001", Severity.INFO,
                   f"{len(outliers)} sinistres > Q3+5×IQR. Ratio max/médiane={ratio:.1f}",
                   metric=ratio)


def _check_dev_year_zero_claims(claims, epi, report):
    """Années à 0 sinistre : potentiellement IBNR à compléter."""
    if 'annee' not in claims.columns:
        return
    years_with_claims = set(claims['annee'].unique())
    years_with_epi = set(epi['annee'].unique())
    zero_years = sorted(years_with_epi - years_with_claims)
    if zero_years:
        report.add("ZERO_001", Severity.INFO,
                   f"Années EPI sans sinistre : {zero_years}. "
                   f"Vérifier IBNR pour années récentes",
                   metric=len(zero_years))


if __name__ == "__main__":
    import sys
    sys.exit(0)
