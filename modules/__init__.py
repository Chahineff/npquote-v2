"""NPquote v2 — modules de tarification réassurance.

Upgrade P0+P1 de l'outil NPquote (SCR/CatCo 2018) :
- datacheck     : contrôles cohérence inputs courtiers
- compte_technique : LR/COR/Résultat/Capital S2/RoE
- bouquet       : agrégation multi-tranches avec part variable
- aad_aal       : conditions annuelles agrégées (BC + Monte Carlo)
- propquote     : Quote-Part + Surplus avec sliding scale
- triangles     : Chain Ladder + Mack + Bornhuetter-Ferguson + Cape Cod
- gpd_pot       : Generalized Pareto + Peaks Over Threshold + KS/AD
- broker_import : ingestion multi-format avec mapping configurable
"""

__version__ = "2.0.0"
__author__ = "SMGA AME — upgrade NPquote v1 (SCR/CatCo 2018)"
