"""Mode interactif NPquote v2 — pas besoin de JSON.

Pose questions en français, construit config, lance pipeline.
Usage : python3 run_interactif.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.auto_broker_parser import parse_renewal_folder


FX_DEFAULTS = {
    "EGP": 0.021, "EUR": 1.08, "USD": 1.00, "GBP": 1.27,
    "MAD": 0.10, "AED": 0.272, "MGA": 0.00022, "TND": 0.32,
    "NGN": 0.00065, "ZAR": 0.054, "XOF": 0.0016, "XAF": 0.0016,
}


def ask(question: str, default: str = None, choices: list = None) -> str:
    """Pose question avec défaut + choix optionnels."""
    suffix = ""
    if choices:
        suffix = f" [{'/'.join(choices)}]"
    if default is not None:
        suffix += f" (défaut={default})"
    while True:
        rep = input(f"  ➤ {question}{suffix} : ").strip()
        if not rep and default is not None:
            return default
        if choices and rep.lower() not in [c.lower() for c in choices]:
            print(f"    ⚠ Choix invalide. Options: {choices}")
            continue
        if rep:
            return rep


def ask_float(question: str, default: float = None,
              min_val: float = None, max_val: float = None) -> float:
    """Pose question numérique avec validation."""
    while True:
        rep = ask(question, default=str(default) if default is not None else None)
        try:
            val = float(rep.replace(',', '.').replace('%', ''))
            if min_val is not None and val < min_val:
                print(f"    ⚠ Doit être >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    ⚠ Doit être <= {max_val}")
                continue
            return val
        except ValueError:
            print(f"    ⚠ Pas un nombre valide")


def ask_yes_no(question: str, default: str = "n") -> bool:
    """Question oui/non."""
    rep = ask(question, default=default, choices=["o", "n"])
    return rep.lower() in ('o', 'oui', 'y', 'yes')


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_box(lines: list[str]):
    width = max(len(l) for l in lines) + 4
    print(f"  ╔{'═'*width}╗")
    for l in lines:
        print(f"  ║  {l:<{width-4}}  ║")
    print(f"  ╚{'═'*width}╝")


def main():
    print_section("NPquote v2 — Mode Interactif")
    print("  Tu réponds aux questions. Pas besoin de JSON.")
    print("  Appuie [Entrée] pour garder valeur par défaut.\n")

    # === Étape 1 : Dossier ===
    print_section("1/5 — Dossier des renewal packs")
    while True:
        folder = ask("Chemin dossier (glisse depuis Finder ou tape chemin)",
                     default=str(ROOT.parent))
        folder = folder.strip("'\"").strip()
        if Path(folder).exists() and Path(folder).is_dir():
            break
        print(f"    ⚠ Dossier introuvable : {folder}")

    # === Étape 2 : Cedante + devise ===
    print_section("2/5 — Cédante + Devise")
    cedante = ask("Nom de la cédante", default="Unknown")
    print(f"\n  Devises connues : {', '.join(FX_DEFAULTS.keys())}")
    currency = ask("Devise du programme", default="EGP").upper()
    fx_default = FX_DEFAULTS.get(currency, 1.0)
    if currency not in FX_DEFAULTS:
        fx = ask_float(f"Taux 1 {currency} = ? USD", default=fx_default)
    else:
        keep = ask_yes_no(f"FX {currency}→USD = {fx_default}. Garder ?",
                          default="o")
        fx = fx_default if keep else ask_float("Nouveau taux", default=fx_default)
    annee = int(ask_float("Année du renouvellement", default=datetime.now().year + 1))

    # === Étape 3 : Scan + détection traités ===
    print_section("3/5 — Scan du dossier + détection traités")
    print(f"  Scanning {folder}...")
    pack = parse_renewal_folder(Path(folder), cedante=cedante, currency=currency)
    print(f"\n  ✓ {pack.detection_report['n_files']} fichiers scannés")
    print(f"  ✓ {len(pack.treaties)} traités détectés")
    print(f"  ✓ {len(pack.epi_by_lob)} LOB avec EPI")
    print(f"  ✓ {len(pack.xol_layers)} couches XOL")

    if not pack.treaties:
        print("\n  ⚠ Aucun traité trouvé. Vérifier dossier.")
        return

    print("\n  Traités détectés :")
    treaty_keys = []
    for i, t in enumerate(pack.treaties):
        key = f"{t.lob}_{t.treaty_type}"
        treaty_keys.append((key, t))
        # Calc prime estimée
        epi = pack.epi_by_lob.get(t.lob, {}).get('next', {})
        if t.treaty_type == 'qs':
            prime_100 = epi.get('qs_cession', 0)
        elif t.treaty_type == 'surplus':
            prime_100 = epi.get('surplus', 0)
        else:
            prime_100 = t.by_uy['premium'].tail(2).mean() if not t.by_uy.empty else 0
        prime_usd = prime_100 * fx
        # Calc LR moyen
        lrs = t.by_uy['loss_ratio'].values
        lr_avg = lrs.mean() if len(lrs) else 0
        flag = "🟢" if lr_avg < 0.50 else ("🟡" if lr_avg < 0.80 else "🔴")
        print(f"    [{i+1}] {flag} {key:35s} LR_avg={lr_avg:>6.1%}  "
              f"Prime 100%={prime_100:>13,.0f} {currency} "
              f"(~{prime_usd:>9,.0f} USD)")

    # === Étape 4 : Mode + parts ===
    print_section("4/5 — Stratégie de souscription")
    print("  3 modes disponibles :")
    print("    [1] Uniforme — même part% sur tous les traités")
    print("    [2] Variable — part% différente par traité")
    print("    [3] Capacité USD — tu fixes le montant USD max par traité")
    mode = ask("Choix", default="1", choices=["1", "2", "3"])

    config = {
        "cedante": cedante, "currency": currency, "annee": annee,
        "fx_to_usd": {currency: fx, "USD": 1.0},
    }

    if mode == "1":
        share = ask_float("Part % uniforme (ex: 10 pour 10%)",
                          default=10, min_val=0, max_val=100) / 100
        config["uniform_share"] = share
        print(f"\n  → Mode UNIFORME {share:.0%} sur {len(treaty_keys)} traités")

    elif mode == "2":
        print("\n  Pour chaque traité, tape la part% (0 = exclure).")
        print("  Recommandation visible : LR_avg historique.")
        shares = {}
        for key, t in treaty_keys:
            lrs = t.by_uy['loss_ratio'].values
            lr_avg = lrs.mean() if len(lrs) else 0
            # Suggestion auto basée sur LR
            if lr_avg < 0.30:
                suggest = 25
            elif lr_avg < 0.60:
                suggest = 15
            elif lr_avg < 0.90:
                suggest = 5
            else:
                suggest = 0
            share_pct = ask_float(f"{key} (LR_hist={lr_avg:.1%}, suggéré={suggest}%)",
                                   default=suggest, min_val=0, max_val=100) / 100
            if share_pct > 0:
                shares[key] = share_pct
        config["shares"] = shares
        config["default_share"] = 0.0

    else:  # USD capacity
        print("\n  Pour chaque traité, tape capacité USD max (0 = exclure).")
        usd_caps = {}
        for key, t in treaty_keys:
            cap = ask_float(f"Capacité USD pour {key}",
                            default=0, min_val=0)
            if cap > 0:
                usd_caps[key] = cap
        config["usd_capacity"] = usd_caps
        config["default_share"] = 0.0

    # === Étape 5 : Seuils décision ===
    print_section("5/5 — Seuils décision")
    target_roe = ask_float("RoE cible (ex: 12 pour 12%)",
                            default=12, min_val=0, max_val=50) / 100
    max_cor = ask_float("COR max acceptable (ex: 95 pour 95%)",
                         default=95, min_val=50, max_val=200) / 100
    market_lr = ask_float("LR a priori marché (Bühlmann credibility)",
                          default=65, min_val=0, max_val=200) / 100

    config["target_roe"] = target_roe
    config["max_combined_ratio"] = max_cor
    config["market_lr_priori"] = market_lr

    # Sauvegarde config (audit + réutilisation)
    config_dir = ROOT / "configs"
    config_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_dir / f"interactif_{cedante.replace(' ', '_')}_{ts}.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n  ✓ Config sauvegardée : {config_path.name}")

    # === Récap avant lancement ===
    print_section("RÉCAPITULATIF")
    print_box([
        f"Cédante       : {cedante}",
        f"Devise        : {currency} (1 {currency} = {fx} USD)",
        f"Année         : {annee}",
        f"Mode          : {['Uniforme', 'Variable', 'USD capacity'][int(mode)-1]}",
        f"RoE cible     : {target_roe:.0%}",
        f"COR max       : {max_cor:.0%}",
        f"LR marché     : {market_lr:.0%}",
    ])

    if not ask_yes_no("\n  Lancer l'analyse maintenant ?", default="o"):
        print("\n  Annulé. Config sauvegardée pour relancer plus tard avec :")
        print(f"  python3 run_renewal.py --folder {folder} --config {config_path}")
        return

    # === Lancement ===
    output_dir = ask("Dossier de sortie",
                     default=str(Path.home() / "Desktop"))
    output_dir = Path(output_dir.strip("'\"").strip())
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("EXÉCUTION")
    from run_renewal import run
    result = run(
        folder=folder, config_path=str(config_path),
        cedante=cedante, currency=currency,
        fx_to_usd=config["fx_to_usd"],
        output_dir=str(output_dir),
    )

    print_section("TERMINÉ")
    print(f"  Workbook   : {result['output_file']}")
    print(f"  Décision   : {result['decision']['color']} {result['decision']['verdict']}")
    print(f"  Loss Ratio : {result['bouquet']['loss_ratio']:.1%}")
    print(f"  COR        : {result['bouquet']['combined_ratio']:.1%}")
    print(f"  RoE        : {result['bouquet']['roe_diversified']:.1%}")
    print(f"  EVA        : {result['bouquet']['eva_bouquet']:,.0f} {currency}")

    if ask_yes_no("\n  Ouvrir le workbook maintenant ?", default="o"):
        import subprocess
        subprocess.run(["open", result['output_file']])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Annulé par utilisateur.")
        sys.exit(0)
