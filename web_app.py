"""NPquote v2 — Web App Streamlit.

Lancement local : streamlit run web_app.py
Deploy cloud   : push GitHub → connect Streamlit Cloud
"""
import sys
from pathlib import Path
import tempfile
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.auto_broker_parser import parse_renewal_folder
from run_renewal import (
    determine_shares, compute_treaty_metrics,
    aggregate_bouquet_with_copula, make_decision,
    map_lob_to_s2,
)
from modules.raroc import compute_full_scr
from modules.versioning import save_quote
from modules.uw_sheet import (
    TreatyHeader, build_uw_sheet_from_pack, add_xol_layers_from_pack,
    prop_to_dataframe, nonprop_to_dataframe, write_uw_sheet_to_excel,
    uw_sheet_to_html,
)
from modules.wording_clauses import (
    get_standard_package, list_all_clauses, TreatyType,
)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPquote v2 — SMGA AME",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

FX_DEFAULTS = {
    "EGP": 0.021, "EUR": 1.08, "USD": 1.00, "GBP": 1.27,
    "MAD": 0.10, "AED": 0.272, "MGA": 0.00022, "TND": 0.32,
    "NGN": 0.00065, "ZAR": 0.054, "XOF": 0.0016, "XAF": 0.0016,
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 NPquote v2")
st.sidebar.markdown("**SMGA AME — Renewal Analyzer**")
st.sidebar.divider()
st.sidebar.markdown("""
**Pipeline** :
1. Upload renewal pack
2. Configure stratégie
3. Run analyse
4. Download workbook
""")
st.sidebar.divider()
st.sidebar.caption(f"v2.0.0 — {datetime.now().year}")


# ─── Main header ──────────────────────────────────────────────────────────────
st.title("NPquote v2 — Reinsurance Renewal Analyzer")
st.caption("Upload renewal pack → choose share strategy → get decision + workbook")


# ─── Étape 1 : Upload ─────────────────────────────────────────────────────────
st.header("1️⃣ Upload Renewal Pack")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Drag-drop fichiers courtier (.xlsx, .xlsm, .xls, .ods, .csv)",
        type=["xlsx", "xlsm", "xls", "ods", "csv"],
        accept_multiple_files=True,
        help="Multi-format : Excel (xlsx/xlsm/xls), LibreOffice (ods), CSV.",
    )

with col2:
    use_demo = st.button("🧪 Utiliser dataset démo (Al Wataniya)",
                          use_container_width=True)
    if use_demo:
        st.session_state['use_demo'] = True


# ─── Stage uploaded files to temp folder ──────────────────────────────────────
tmp_folder = None
if uploaded_files:
    tmp_folder = Path(tempfile.mkdtemp(prefix="npquote_upload_"))
    for uf in uploaded_files:
        (tmp_folder / uf.name).write_bytes(uf.read())
    st.success(f"✅ {len(uploaded_files)} fichiers uploadés")
elif st.session_state.get('use_demo'):
    tmp_folder = Path("/tmp/al_wataniya_pack")
    if tmp_folder.exists():
        st.info("📦 Dataset démo Al Wataniya chargé")
    else:
        st.error("Dataset démo introuvable. Upload tes propres fichiers.")
        tmp_folder = None


# ─── Étape 2 : Config cedante ─────────────────────────────────────────────────
if tmp_folder:
    st.header("2️⃣ Configuration Cédante")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cedante = st.text_input("Nom cédante", value="Al Wataniya Insurance")
    with col2:
        currency = st.selectbox("Devise", options=list(FX_DEFAULTS.keys()), index=0)
    with col3:
        fx = st.number_input(f"Taux 1 {currency} → USD",
                              value=FX_DEFAULTS[currency], format="%.4f")
    with col4:
        annee = st.number_input("Année", min_value=2020, max_value=2035,
                                  value=2026, step=1)

    # Champs UW Sheet supplémentaires
    with st.expander("📋 Champs UW Sheet (slip professionnel)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            broker = st.text_input("Broker", value="")
            inception = st.text_input("Inception Date", value=f"{annee}-01-01")
        with col2:
            leader = st.text_input("Leader Reinsurer", value="SMGA AME")
            expiry = st.text_input("Expiry Date", value=f"{annee}-12-31")
        with col3:
            country = st.text_input("Country / Territory", value="")
            underwriter = st.text_input("Underwriter (you)", value="")
        period_basis = st.selectbox("Period basis",
                                      options=["RAD (Risks Attaching During)",
                                                "LOD (Losses Occurring During)"],
                                      index=0)
        business_class = st.text_input("Business class",
                                         value="All classes (treaty whole account)")


    # ─── Étape 3 : Scan auto ──────────────────────────────────────────────
    st.header("3️⃣ Scan + Détection Auto")
    try:
        with st.spinner("Scanning renewal pack..."):
            pack = parse_renewal_folder(tmp_folder, cedante=cedante, currency=currency)
    except Exception as e:
        st.error(f"Erreur scan : {e}")
        st.info("Vérifie : fichiers ouvrables, pas corrompus, format Excel/ODS/CSV valide.")
        with st.expander("Détails techniques"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fichiers", pack.detection_report['n_files'])
    col2.metric("Traités", len(pack.treaties))
    col3.metric("LOB avec EPI", len(pack.epi_by_lob))
    col4.metric("Couches XOL", len(pack.xol_layers))

    if not pack.treaties:
        st.warning("⚠ Aucun traité QS/Surplus/XOL détecté automatiquement.")
        with st.expander("Voir fichiers scannés"):
            for fname, finfo in pack.detection_report['files'].items():
                st.write(f"**{fname}**")
                for sh in finfo.get('sheets', []):
                    st.caption(f"  • {sh['name']} (type={sh['type']}, LOB={sh['lob']})")
        st.info("Tip : nomme les feuilles avec mots-clés QS/Surplus/Statistics/XOL "
                 "pour détection auto. Sinon utilise format Al Wataniya (cf demo).")
        st.stop()

    # Tableau traités détectés
    treaties_summary = []
    for t in pack.treaties:
        key = f"{t.lob}_{t.treaty_type}"
        epi = pack.epi_by_lob.get(t.lob, {}).get('next', {})
        if t.treaty_type == 'qs':
            prime_100 = epi.get('qs_cession', 0)
        elif t.treaty_type == 'surplus':
            prime_100 = epi.get('surplus', 0)
        else:
            prime_100 = t.by_uy['premium'].tail(2).mean() if not t.by_uy.empty else 0
        lrs = t.by_uy['loss_ratio'].values
        lr_avg = lrs.mean() if len(lrs) else 0
        flag = "🟢" if lr_avg < 0.50 else ("🟡" if lr_avg < 0.80 else "🔴")
        treaties_summary.append({
            "Traité": f"{flag} {key}",
            "Key": key,
            "LR moyen": f"{lr_avg:.1%}",
            "Lr_num": lr_avg,
            f"Prime 100% ({currency})": f"{prime_100:,.0f}",
            "Prime USD": f"{prime_100 * fx:,.0f}",
            "Prime_local_num": prime_100,
        })
    df_treaties = pd.DataFrame(treaties_summary)
    st.dataframe(df_treaties[["Traité", "LR moyen",
                               f"Prime 100% ({currency})", "Prime USD"]],
                  use_container_width=True, hide_index=True)


    # ─── Étape 4 : Stratégie ──────────────────────────────────────────────
    st.header("4️⃣ Stratégie de Souscription")
    mode = st.radio(
        "Mode",
        options=["Uniforme", "Variable par traité", "Capacité USD par traité"],
        horizontal=True,
        help="Uniforme = même %. Variable = % différent par traité. USD = montant absolu.",
    )

    config = {
        "cedante": cedante, "currency": currency, "annee": int(annee),
        "fx_to_usd": {currency: fx, "USD": 1.0},
    }

    if mode == "Uniforme":
        uniform = st.slider("Part uniforme (%)", 0, 100, 10, step=1) / 100
        config["uniform_share"] = uniform

    elif mode == "Variable par traité":
        st.markdown("**Ajuste % par traité. Suggestion auto basée sur LR historique.**")
        shares = {}
        for row in treaties_summary:
            lr = row["Lr_num"]
            if lr < 0.30:
                suggest = 25
            elif lr < 0.60:
                suggest = 15
            elif lr < 0.90:
                suggest = 5
            else:
                suggest = 0
            share_pct = st.slider(
                f"{row['Traité']} (LR={row['LR moyen']})",
                0, 50, suggest, step=1,
                key=f"share_{row['Key']}",
            ) / 100
            if share_pct > 0:
                shares[row["Key"]] = share_pct
        config["shares"] = shares
        config["default_share"] = 0.0

    else:  # Capacité USD
        st.markdown("**Fixe capacité USD max par traité. Système calcule part%.**")
        usd_caps = {}
        for row in treaties_summary:
            cap = st.number_input(
                f"{row['Traité']} (Prime 100% = {row['Prime USD']} USD)",
                min_value=0, max_value=1_000_000,
                value=0, step=1000,
                key=f"usd_{row['Key']}",
            )
            if cap > 0:
                usd_caps[row["Key"]] = cap
        config["usd_capacity"] = usd_caps
        config["default_share"] = 0.0


    # ─── Étape 5 : Seuils décision ────────────────────────────────────────
    st.header("5️⃣ Seuils Décision")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_roe = st.slider("RoE cible (%)", 0, 30, 12) / 100
    with col2:
        max_cor = st.slider("COR max (%)", 50, 130, 95) / 100
    with col3:
        market_lr = st.slider("LR a priori marché (Bühlmann)", 0, 100, 65) / 100
    config["target_roe"] = target_roe
    config["max_combined_ratio"] = max_cor
    config["market_lr_priori"] = market_lr


    # ─── Run ──────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            shares_resolved = determine_shares(pack, config, config["fx_to_usd"])

            # Treaty metrics
            treaty_metrics = []
            historical_lrs = {}
            for key, info in shares_resolved.items():
                if info["share_pct"] <= 0:
                    continue
                m = compute_treaty_metrics(
                    info["treaty"], info["share_pct"], info["prime_100_local"],
                    market_lr_priori=market_lr,
                )
                if m:
                    treaty_metrics.append(m)
                    historical_lrs[key] = info["treaty"].by_uy['loss_ratio'].values

            if not treaty_metrics:
                st.error("⚠ Aucune part > 0. Augmente parts pour analyser.")
                st.stop()

            bouquet = aggregate_bouquet_with_copula(treaty_metrics, historical_lrs)

            # SCR
            premiums_by_lob = {}
            for m in treaty_metrics:
                s2_lob = map_lob_to_s2(m['lob_treaty'].split(' ')[0].lower(),
                                        m['lob_treaty'].split(' ')[1].lower())
                premiums_by_lob[s2_lob] = premiums_by_lob.get(s2_lob, 0) + m['prime_share_local']
            reserves_by_lob = {k: v * 0.5 for k, v in premiums_by_lob.items()}
            pml = {"manmade_fire": bouquet['total_prime'] * 2.0}
            scr = compute_full_scr(premiums_by_lob, reserves_by_lob, pml)
            decision = make_decision(bouquet, target_roe=target_roe, max_cor=max_cor)

            # Save version
            snap = save_quote(
                cedante=cedante, annee=int(annee),
                inputs={"config": config},
                outputs={**bouquet, "scr_total": scr.scr_total},
                decision=decision["verdict"],
                decision_reasons=decision["reasons"],
            )

        # ─── Résultats ───
        st.success(f"✅ Analyse terminée. Snapshot : `{snap.quote_id}`")

        # Décision banner
        verdict_color = {"🟢": "success", "🟡": "warning", "🔴": "error"}.get(
            decision["color"], "info")
        getattr(st, verdict_color)(
            f"## {decision['color']} **{decision['verdict']}**\n\n"
            + ("\n".join(f"- {r}" for r in decision['reasons'])
                if decision['reasons'] else "Tous critères OK.")
        )

        # KPIs principaux
        st.subheader("KPIs Bouquet")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prime totale (USD)",
                     f"{bouquet['total_prime']*fx:,.0f}",
                     help=f"{bouquet['total_prime']:,.0f} {currency}")
        col2.metric("Loss Ratio", f"{bouquet['loss_ratio']:.1%}")
        col3.metric("Combined Ratio", f"{bouquet['combined_ratio']:.1%}",
                     delta=f"{(bouquet['combined_ratio']-max_cor)*100:+.1f}pp vs max",
                     delta_color="inverse")
        col4.metric("RoE diversifié", f"{bouquet['roe_diversified']:.1%}",
                     delta=f"{(bouquet['roe_diversified']-target_roe)*100:+.1f}pp vs cible")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Résultat technique",
                     f"{bouquet['total_resultat']:,.0f} {currency}")
        col2.metric("EVA",
                     f"{bouquet['eva_bouquet']:,.0f} {currency}",
                     delta_color="normal" if bouquet['eva_bouquet'] >= 0 else "inverse")
        col3.metric("Capital S2 diversifié",
                     f"{bouquet['capital_s2_diversified']:,.0f} {currency}")
        col4.metric("Diversification benefit",
                     f"{bouquet['diversification_benefit_pct']:.1%}")

        # Charts
        st.subheader("📊 Visualisations")
        col1, col2 = st.columns(2)

        with col1:
            df_m = pd.DataFrame(treaty_metrics)
            fig = px.bar(df_m, x="lob_treaty", y="raroc",
                          color="raroc", color_continuous_scale="RdYlGn",
                          title="RAROC par traité",
                          labels={"raroc": "RAROC", "lob_treaty": "Traité"})
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df_m, x="loss_ratio", y="combined_ratio",
                               size="prime_share_local", color="raroc",
                               hover_name="lob_treaty",
                               color_continuous_scale="RdYlGn",
                               title="Loss Ratio vs Combined Ratio (taille = prime)")
            fig2.add_hline(y=max_cor, line_dash="dash", line_color="red",
                            annotation_text=f"COR max {max_cor:.0%}")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # SCR breakdown
        st.subheader("🛡️ SCR Solvency II")
        scr_breakdown = pd.DataFrame({
            "Composante": ["NL Premium+Reserve", "NL Cat", "NL Total",
                            "BSCR", "Op Risk", "SCR Total"],
            "Valeur": [scr.scr_nl_prem_res, scr.scr_nl_cat, scr.scr_nl_total,
                        scr.bscr, scr.operational_risk, scr.scr_total],
        })
        fig3 = px.bar(scr_breakdown, x="Composante", y="Valeur",
                       title=f"Décomposition SCR ({currency})")
        st.plotly_chart(fig3, use_container_width=True)

        # Detail table
        st.subheader("📋 Détail par traité")
        df_show = df_m[["lob_treaty", "share_pct", "lr_credible_buhlmann",
                          "z_credibility", "combined_ratio", "raroc",
                          "eva", "prime_share_local"]].copy()
        df_show.columns = ["Traité", "Part %", "LR crédible", "Z Bühlmann",
                            "COR", "RAROC", "EVA", f"Prime ({currency})"]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Generate workbook for download
        from run_renewal import _write_full_workbook
        out_path = Path(tempfile.mkdtemp()) / f"NPquote_v2_{cedante.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        _write_full_workbook(out_path, pack, shares_resolved, treaty_metrics,
                              bouquet, scr, decision, currency, config["fx_to_usd"])

        # ─── UW SHEET professionnel (slip souscripteur) ─────────────────────
        st.divider()
        st.header("📋 UW Sheet — Slip Souscripteur Professionnel")

        uw_header = TreatyHeader(
            company=cedante, broker=broker, leader=leader,
            inception_date=inception, expiry_date=expiry,
            original_currency=currency, exchange_rate_to_usd=fx,
            period_basis=period_basis, business_class=business_class,
            country=country, underwriter=underwriter,
            quote_id=snap.quote_id,
        )
        uw_sheet = build_uw_sheet_from_pack(pack, shares_resolved,
                                              treaty_metrics, uw_header)
        add_xol_layers_from_pack(uw_sheet, pack, share_pct=0.10)

        # Sélecteur clauses
        with st.expander("📜 Wording / Clauses / Exclusions", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                tt_choice = st.selectbox(
                    "Type de traité (pour package standard)",
                    options=["Quote Share / Surplus",
                             "XL Working", "XL Cat", "Whole Account XL"],
                )
            with col2:
                bc = st.text_input("Branche", value="general")

            tt_map = {
                "Quote Share / Surplus": TreatyType.QS,
                "XL Working": TreatyType.XL_WORKING,
                "XL Cat": TreatyType.XL_CAT,
                "Whole Account XL": TreatyType.WHOLE_ACCOUNT_XL,
            }
            pkg = get_standard_package(tt_map[tt_choice], business_class=bc)
            uw_sheet.clauses = pkg["clauses"]
            uw_sheet.exclusions = pkg["exclusions"]
            uw_sheet.warranties = pkg["warranties"]

            st.write(f"**{len(pkg['clauses'])}** clauses + "
                       f"**{len(pkg['exclusions'])}** exclusions + "
                       f"**{len(pkg['warranties'])}** warranties chargées.")
            with st.expander("Voir liste complète"):
                for label, items in [("Clauses", pkg["clauses"]),
                                       ("Exclusions", pkg["exclusions"]),
                                       ("Warranties", pkg["warranties"])]:
                    st.markdown(f"**{label}**")
                    for i, c in enumerate(items, 1):
                        st.markdown(f"  {i}. {c[:200]}...")

        # Affichage tables Prop + Non Prop
        st.subheader("Treaties Proportionals")
        if uw_sheet.prop_lines:
            st.dataframe(prop_to_dataframe(uw_sheet),
                          use_container_width=True, hide_index=True)
        else:
            st.info("Aucun traité proportionnel.")

        st.subheader("Treaties Non-Proportionals")
        if uw_sheet.nonprop_lines:
            st.dataframe(nonprop_to_dataframe(uw_sheet),
                          use_container_width=True, hide_index=True)
        else:
            st.info("Aucun traité non-proportionnel.")

        # Grand total
        gt = uw_sheet.grand_total()
        st.subheader("Grand Total — Our Share")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total EPI Our Share",
                     f"{gt['total_epi_our_share_usd']:,.0f} USD")
        col2.metric("Total Limit Our Share",
                     f"{gt['total_limit_our_share_usd']:,.0f} USD")
        col3.metric("Expected Total Loss",
                     f"{gt['expected_total_loss_usd']:,.0f} USD")

        # Génère UW Sheet XLSX
        uw_path = Path(tempfile.mkdtemp()) / f"UW_Sheet_{cedante.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        write_uw_sheet_to_excel(uw_sheet, str(uw_path), include_clauses=True)

        st.divider()
        st.subheader("📥 Téléchargements")
        col1, col2 = st.columns(2)
        with col1:
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️ Workbook Analyse Complet",
                    data=f.read(), file_name=out_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        with col2:
            with open(uw_path, "rb") as f:
                st.download_button(
                    label="⬇️ UW Sheet (Slip Pro)",
                    data=f.read(), file_name=uw_path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary", use_container_width=True,
                )

else:
    st.info("👆 Upload des fichiers ou utilise le dataset démo pour commencer.")
