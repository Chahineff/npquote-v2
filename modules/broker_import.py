"""P1.8 — Import des données courtiers avec mapping configurable.

Support : CSV, XLSX, XLSM (incluant lecture du NPquote_v1 historique).
Mapping de colonnes flexible pour adapter aux templates Aon, Guy Carpenter,
Willis, Howden, Marsh.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import pandas as pd


# Synonymes connus → colonne canonique
COLUMN_ALIASES = {
    "annee": ["year", "année", "underwriting year", "uy", "ay",
              "accident year", "occurrence year", "loss year", "policy year"],
    "nom": ["claim", "claim name", "insured", "policyholder", "claimant",
            "claim no", "claim number", "claim ref", "policy ref"],
    "cout": ["loss", "amount", "incurred", "paid", "ultimate", "claim amount",
             "gross loss", "loss amount", "incurred loss", "indemnity",
             "montant", "charge", "sinistre"],
    "date_survenance": ["loss date", "occurrence date", "accident date",
                        "date of loss", "dol", "date sinistre", "date_sin"],
    "date_declaration": ["report date", "notification date", "date declaration"],
    "devise": ["currency", "ccy", "iso currency", "monnaie"],
    "epi": ["premium", "gnpi", "gpi", "gross premium", "gwp", "epi",
            "premium income", "subject premium", "assiette", "prime"],
    "nb_polices": ["nb polices", "policy count", "policies", "n policies",
                   "in force", "number of risks"],
    "branche": ["lob", "line of business", "class", "branch", "branche",
                "product"],
    "pays": ["country", "territory", "domicile", "pays"],
}


@dataclass
class ImportResult:
    claims: pd.DataFrame
    epi: pd.DataFrame
    mapping_used: dict
    warnings: list = field(default_factory=list)
    source_file: str = ""


def _normalize(col: str) -> str:
    """Normalise un nom de colonne : minuscule, sans accents, sans ponctuation."""
    c = str(col).lower().strip()
    c = re.sub(r'[éèêë]', 'e', c)
    c = re.sub(r'[àâä]', 'a', c)
    c = re.sub(r'[ôöó]', 'o', c)
    c = re.sub(r'[ùûü]', 'u', c)
    c = re.sub(r'[îï]', 'i', c)
    c = re.sub(r'[ç]', 'c', c)
    c = re.sub(r'[^a-z0-9 ]', ' ', c)
    c = re.sub(r'\s+', ' ', c).strip()
    return c


def auto_map_columns(df: pd.DataFrame, target_keys: list[str]) -> dict:
    """Tente de mapper les colonnes du df vers les clés canoniques."""
    mapping = {}
    cols_norm = {col: _normalize(col) for col in df.columns}
    used = set()

    for key in target_keys:
        aliases = [key] + COLUMN_ALIASES.get(key, [])
        aliases_norm = [_normalize(a) for a in aliases]
        for col, norm in cols_norm.items():
            if col in used:
                continue
            if norm in aliases_norm:
                mapping[key] = col
                used.add(col)
                break
        else:
            # Match partiel
            for col, norm in cols_norm.items():
                if col in used:
                    continue
                if any(a in norm or norm in a for a in aliases_norm if len(a) > 3):
                    mapping[key] = col
                    used.add(col)
                    break
    return mapping


def read_broker_file(
    path: str | Path,
    *,
    sheet_claims: str | None = None,
    sheet_epi: str | None = None,
    explicit_mapping: dict | None = None,
) -> ImportResult:
    """Lit un fichier courtier multi-format et extrait claims + EPI.

    Heuristique :
    - .csv → 1 seule table (claims), EPI à fournir séparément
    - .xlsx/.xlsm → cherche feuilles "claims/sinistres" et "premium/epi"
    """
    path = Path(path)
    warnings = []

    if path.suffix.lower() == '.csv':
        claims = pd.read_csv(path)
        epi = pd.DataFrame()
    elif path.suffix.lower() in ('.xlsx', '.xlsm'):
        xls = pd.ExcelFile(path)
        sheet_claims = sheet_claims or _guess_sheet(xls.sheet_names,
                                                    ['claim', 'sinistr', 'loss'])
        sheet_epi = sheet_epi or _guess_sheet(xls.sheet_names,
                                              ['premium', 'epi', 'prime', 'expo'])
        claims = pd.read_excel(path, sheet_name=sheet_claims) if sheet_claims else pd.DataFrame()
        epi = pd.read_excel(path, sheet_name=sheet_epi) if sheet_epi else pd.DataFrame()
    else:
        raise ValueError(f"Format non supporté : {path.suffix}")

    if explicit_mapping:
        claims_map = {k: v for k, v in explicit_mapping.items() if v in claims.columns}
        epi_map = {k: v for k, v in explicit_mapping.items() if v in epi.columns}
    else:
        claims_map = auto_map_columns(claims, ['annee', 'nom', 'cout', 'devise',
                                                'branche', 'pays'])
        epi_map = auto_map_columns(epi, ['annee', 'epi', 'nb_polices', 'devise',
                                          'branche'])

    claims_clean = _rename_and_clean(claims, claims_map)
    epi_clean = _rename_and_clean(epi, epi_map)

    if 'annee' in claims_clean.columns:
        claims_clean['annee'] = pd.to_numeric(claims_clean['annee'],
                                              errors='coerce').astype('Int64')
    if 'cout' in claims_clean.columns:
        claims_clean['cout'] = pd.to_numeric(claims_clean['cout'], errors='coerce')
    if 'annee' in epi_clean.columns:
        epi_clean['annee'] = pd.to_numeric(epi_clean['annee'],
                                           errors='coerce').astype('Int64')
    if 'epi' in epi_clean.columns:
        epi_clean['epi'] = pd.to_numeric(epi_clean['epi'], errors='coerce')

    if claims.shape[0] > 0 and len(claims_map) < 3:
        warnings.append(f"Mapping claims incomplet : seules {list(claims_map)} "
                        f"mappées sur {list(claims.columns)}")
    if epi.shape[0] > 0 and len(epi_map) < 2:
        warnings.append(f"Mapping EPI incomplet : seules {list(epi_map)} "
                        f"mappées sur {list(epi.columns)}")

    return ImportResult(
        claims=claims_clean, epi=epi_clean,
        mapping_used={"claims": claims_map, "epi": epi_map},
        warnings=warnings, source_file=str(path),
    )


def _guess_sheet(sheet_names: list[str], keywords: list[str]) -> str | None:
    for sn in sheet_names:
        snn = _normalize(sn)
        if any(k in snn for k in keywords):
            return sn
    return None


def _rename_and_clean(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if df.empty:
        return df
    inv = {v: k for k, v in mapping.items()}
    return df.rename(columns=inv)[list(inv.values())].copy()


def read_npquote_v1(path: str | Path) -> ImportResult:
    """Lecteur spécialisé pour le format NPquote_v1.xlsm.

    Lit les feuilles 'Sinistres' et 'EPI' avec leurs structures connues
    (en-têtes à la ligne 4 pour Sinistres, ligne 6 pour EPI).
    """
    path = Path(path)
    # Sinistres : header ligne 5 (index 4)
    claims = pd.read_excel(path, sheet_name='Sinistres', header=4,
                           engine='openpyxl')
    claims = claims.rename(columns={
        'Année': 'annee', 'Nom': 'nom',
        'Coût': 'cout', 'Coût actualisé': 'cout_actualise',
        'Coût utilisé': 'cout_utilise', 'Exclure': 'exclure'
    })
    claims = claims.dropna(subset=['annee', 'cout'])
    if 'exclure' in claims.columns:
        claims = claims[claims['exclure'].isna() | (claims['exclure'] != '*')]

    # EPI : header ligne 8 (index 7)
    epi = pd.read_excel(path, sheet_name='EPI', header=7, engine='openpyxl')
    epi = epi.rename(columns={
        'Année': 'annee', 'EPI': 'epi',
        'Nb de polices': 'nb_polices',
        'EPI actualisé': 'epi_actualise',
    })
    epi = epi[['annee', 'epi', 'nb_polices']].dropna(subset=['annee', 'epi'])
    epi['annee'] = epi['annee'].astype(int)

    return ImportResult(
        claims=claims, epi=epi,
        mapping_used={"claims": "NPquote_v1 native",
                      "epi": "NPquote_v1 native"},
        source_file=str(path),
    )


def standardize_currency(df: pd.DataFrame, fx_rates: dict[str, float],
                         target: str, col_amount: str = 'cout') -> pd.DataFrame:
    """Convertit toutes les valeurs en devise cible avec taux FX donnés.

    fx_rates : {'EUR': 1.0, 'USD': 0.92, 'MAD': 0.092, ...} (vers target=EUR)
    """
    if 'devise' not in df.columns:
        return df.copy()
    df = df.copy()
    df[f"{col_amount}_orig"] = df[col_amount]
    df[f"{col_amount}"] = df.apply(
        lambda row: row[col_amount] * fx_rates.get(row['devise'], 1.0)
                    if pd.notna(row[col_amount]) else row[col_amount],
        axis=1
    )
    df['devise_orig'] = df['devise']
    df['devise'] = target
    return df


def build_triangle_from_claims(
    claims: pd.DataFrame,
    *,
    valuation_dates: list[int] | None = None,
    year_col: str = 'annee',
    amount_col: str = 'cout',
    valuation_col: str = 'date_evaluation',
) -> pd.DataFrame:
    """Construit un triangle cumulé à partir d'une liste sinistre avec dates.

    Si les valuation_dates ne sont pas fournies, déduit depuis valuation_col.
    """
    if valuation_col not in claims.columns:
        # Triangle dégénéré : 1 colonne par année
        return claims.groupby(year_col)[amount_col].sum().to_frame('cumul')

    valuations = valuation_dates or sorted(claims[valuation_col].dropna().unique())
    origins = sorted(claims[year_col].dropna().unique())

    matrix = []
    for o in origins:
        row = []
        for v in valuations:
            mask = (claims[year_col] == o) & (claims[valuation_col] <= v)
            row.append(claims.loc[mask, amount_col].sum() if mask.any() else np.nan)
        matrix.append(row)
    delays = [v - origins[0] for v in valuations] if valuations else []
    return pd.DataFrame(matrix, index=origins, columns=delays)
