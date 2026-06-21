"""P3.2 — Bibliothèque de clauses / wording standard pour slip réassurance.

Standard market clauses utilisées par souscripteurs Tier-1 :
- NMA (Nuclear / War / Marine)
- Hours clause (24/72/168/504/720 selon péril)
- Terrorism exclusions (NMA2918, LMA3030, etc.)
- Sanctions (LMA3100/LMA3200)
- Cyber (CL380, NMA2914)
- Communicable Disease (LMA5391, LMA5393)
- Service of Suit / Arbitration
- Currency clauses
- Errors & Omissions
- Insolvency

Référence : Lloyd's Market Association (LMA), International Underwriting
Association (IUA), Munich Re wordings library.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class TreatyType(Enum):
    QS = "quote_share"
    SURPLUS = "surplus"
    FACILITY = "facility"
    XL_WORKING = "xl_working"
    XL_CAT = "xl_cat"
    STOP_LOSS = "stop_loss"
    WHOLE_ACCOUNT_XL = "whole_account_xl"


# ─── Clauses standard ─────────────────────────────────────────────────────────

STANDARD_CLAUSES = {
    "HOURS_72_PROPERTY": (
        "Hours Clause (72 hours - Property): All loss occurrences arising out of "
        "any one Event shall be deemed to constitute one occurrence, provided that "
        "all losses are sustained within a continuous period of 72 hours. The "
        "Reinsured shall have the right to elect the moment from which the said "
        "72-hour period shall commence."
    ),
    "HOURS_168_FLOOD": (
        "Hours Clause (168 hours - Flood / Storm): For floods, storms, and "
        "hurricanes, the period shall be 168 consecutive hours."
    ),
    "HOURS_504_EQ": (
        "Hours Clause (504 hours - Earthquake): For earthquake, the period shall "
        "be 504 consecutive hours (21 days)."
    ),
    "HOURS_720_BUSHFIRE": (
        "Hours Clause (720 hours - Bushfire): For bushfire / wildfire, the period "
        "shall be 720 consecutive hours (30 days)."
    ),
    "PREMIUM_PAYMENT_60D": (
        "Premium Payment Clause: Premiums shall be paid within 60 days of the "
        "inception date or as otherwise agreed. Failure to pay may result in "
        "cancellation."
    ),
    "REINSTATEMENT_PRO_RATA": (
        "Reinstatement of Cover: In the event of loss, the cover shall be "
        "automatically reinstated for the unexpired portion of the period, the "
        "reinstatement premium being calculated pro-rata temporis and pro-rata "
        "amount at the same rate as the original premium."
    ),
    "ARBITRATION_ICC": (
        "Arbitration Clause: All disputes arising out of or in connection with this "
        "Treaty shall be finally settled under the Rules of Arbitration of the "
        "International Chamber of Commerce by one or more arbitrators appointed in "
        "accordance with said Rules."
    ),
    "GOVERNING_LAW_ENGLISH": (
        "Governing Law: This Treaty shall be governed by and construed in accordance "
        "with English law."
    ),
    "CURRENCY_USD": (
        "Currency Clause: All amounts under this Treaty shall be paid in USD. Where "
        "amounts are originally expressed in another currency, conversion shall be "
        "made at the exchange rate prevailing on the date of payment."
    ),
    "ERRORS_OMISSIONS": (
        "Errors and Omissions Clause: Any inadvertent error, omission, or oversight "
        "by either party shall not prejudice the rights of either party hereunder, "
        "provided that such error or omission is rectified as soon as possible "
        "after discovery."
    ),
    "ACCESS_TO_RECORDS": (
        "Access to Records: The Reinsurer shall have the right to inspect, through "
        "its authorised representatives, all books and records of the Reinsured "
        "relating to business covered under this Treaty, at any reasonable time "
        "during or after the term of this Treaty."
    ),
    "ULTIMATE_NET_LOSS": (
        "Ultimate Net Loss: The actual loss sustained by the Reinsured, including "
        "all loss adjustment expenses (excluding office expenses and salaries of "
        "regular employees), after all salvages, subrogations, and recoveries from "
        "other reinsurances."
    ),
    "FOLLOW_THE_FORTUNES": (
        "Follow the Fortunes / Settlements Clause: The Reinsurer shall follow the "
        "settlements of the Reinsured in all matters falling within the terms of "
        "this Treaty, provided such settlements are within the conditions and "
        "limits of the original policies and of this Treaty."
    ),
    "INSOLVENCY": (
        "Insolvency Clause: In the event of insolvency of the Reinsured, "
        "reinsurance under this Treaty shall be payable directly to the Reinsured "
        "or its liquidator, receiver, or statutory successor."
    ),
}


# ─── Exclusions standard ──────────────────────────────────────────────────────

STANDARD_EXCLUSIONS = {
    "NMA1975_WAR": (
        "NMA 1975 — War & Civil War Exclusion: This Treaty excludes loss or "
        "damage directly or indirectly occasioned by, happening through or in "
        "consequence of war, invasion, acts of foreign enemies, hostilities "
        "(whether war be declared or not), civil war, rebellion, revolution, "
        "insurrection, military or usurped power."
    ),
    "NMA1331_NUCLEAR": (
        "NMA 1331 — Nuclear Energy Risks Exclusion: This Treaty excludes "
        "ionising radiation from or contamination by radioactivity from any "
        "nuclear fuel or from any nuclear waste from the combustion of nuclear fuel."
    ),
    "NMA2918_TERRORISM": (
        "NMA 2918 — Terrorism Exclusion: This Treaty excludes loss, damage, cost "
        "or expense of whatsoever nature directly or indirectly caused by, "
        "resulting from or in connection with any act of terrorism regardless of "
        "any other cause or event contributing concurrently or in any other "
        "sequence to the loss."
    ),
    "LMA3030_BIOLOGICAL": (
        "LMA 3030 — Chemical, Biological, Bio-chemical, and Electromagnetic "
        "Weapons Exclusion: This Treaty excludes loss caused by chemical, "
        "biological, bio-chemical, or electromagnetic weapons."
    ),
    "LMA3100_SANCTIONS": (
        "LMA 3100 — Sanction Limitation and Exclusion Clause: No reinsurer shall "
        "be deemed to provide cover and no reinsurer shall be liable to pay any "
        "claim or provide any benefit hereunder to the extent that the provision "
        "of such cover, payment of such claim or provision of such benefit would "
        "expose that reinsurer to any sanction, prohibition or restriction under "
        "United Nations resolutions or the trade or economic sanctions, laws or "
        "regulations of the European Union, United Kingdom or United States of "
        "America."
    ),
    "CL380_CYBER": (
        "CL 380 — Institute Cyber Attack Exclusion Clause: Subject only to "
        "specified exceptions, this Treaty excludes loss damage liability or "
        "expense directly or indirectly caused by or contributed to by or arising "
        "from the use or operation, as a means for inflicting harm, of any "
        "computer, computer system, computer software programme, malicious code, "
        "computer virus or process or any other electronic system."
    ),
    "LMA5391_DISEASE": (
        "LMA 5391 — Communicable Disease Exclusion: This Treaty excludes all "
        "actual or alleged loss, liability, damage, compensation, injury, sickness, "
        "disease, death, medical payment, defence cost, cost, expense or any other "
        "amount, directly or indirectly arising out of, attributable to, or "
        "occurring concurrently or in any sequence with a Communicable Disease."
    ),
    "POLLUTION": (
        "Pollution and Contamination Exclusion (Property Treaty): This Treaty "
        "excludes loss or damage caused by seepage, pollution or contamination, "
        "unless caused by a sudden, identifiable, unintended and unexpected "
        "happening which takes place in its entirety at a specific time and place "
        "during the period of this Treaty."
    ),
    "ASBESTOS": (
        "Asbestos Exclusion: This Treaty excludes all claims arising from "
        "asbestos, asbestos fibres or asbestos derivatives."
    ),
}


# ─── Warranties standard ──────────────────────────────────────────────────────

STANDARD_WARRANTIES = {
    "AGGREGATE_LIMIT": (
        "Aggregate Limit Warranty: The Reinsured warrants that the maximum any one "
        "risk and the aggregate exposure shall not exceed the limits stated in the "
        "Treaty Schedule."
    ),
    "ORIGINAL_RATES": (
        "Original Rating Warranty: The Reinsured warrants that the rates charged "
        "for the original policies shall not be less than the technical minimum "
        "rates communicated to the Reinsurer."
    ),
    "PML_ASSESSMENT": (
        "PML Assessment: The Reinsured warrants that PML / EML assessments shall "
        "be conducted by qualified risk surveyors on all risks with sum insured "
        "exceeding USD 5 million."
    ),
    "PROTECTION_FIRE": (
        "Fire Protection Warranty: The Reinsured warrants that all insured "
        "industrial risks above USD 10 million sum insured shall be protected by "
        "automatic sprinkler systems compliant with NFPA 13 / FM standards."
    ),
}


def get_standard_package(treaty_type: TreatyType, business_class: str = "general"
                          ) -> dict:
    """Retourne le package standard de clauses/exclusions/warranties pour un type de traité.

    business_class : 'fire' / 'engineering' / 'marine' / 'motor' / 'general'
    """
    pkg = {"clauses": [], "exclusions": [], "warranties": []}

    # Base communes à tous traités
    pkg["clauses"].extend([
        STANDARD_CLAUSES["PREMIUM_PAYMENT_60D"],
        STANDARD_CLAUSES["ARBITRATION_ICC"],
        STANDARD_CLAUSES["GOVERNING_LAW_ENGLISH"],
        STANDARD_CLAUSES["CURRENCY_USD"],
        STANDARD_CLAUSES["ERRORS_OMISSIONS"],
        STANDARD_CLAUSES["ACCESS_TO_RECORDS"],
        STANDARD_CLAUSES["FOLLOW_THE_FORTUNES"],
        STANDARD_CLAUSES["INSOLVENCY"],
    ])

    pkg["exclusions"].extend([
        STANDARD_EXCLUSIONS["NMA1975_WAR"],
        STANDARD_EXCLUSIONS["NMA1331_NUCLEAR"],
        STANDARD_EXCLUSIONS["NMA2918_TERRORISM"],
        STANDARD_EXCLUSIONS["LMA3030_BIOLOGICAL"],
        STANDARD_EXCLUSIONS["LMA3100_SANCTIONS"],
        STANDARD_EXCLUSIONS["CL380_CYBER"],
        STANDARD_EXCLUSIONS["LMA5391_DISEASE"],
    ])

    # Spécifiques selon type traité
    if treaty_type in (TreatyType.XL_WORKING, TreatyType.XL_CAT,
                        TreatyType.WHOLE_ACCOUNT_XL):
        pkg["clauses"].append(STANDARD_CLAUSES["HOURS_72_PROPERTY"])
        pkg["clauses"].append(STANDARD_CLAUSES["HOURS_504_EQ"])
        pkg["clauses"].append(STANDARD_CLAUSES["HOURS_168_FLOOD"])
        pkg["clauses"].append(STANDARD_CLAUSES["REINSTATEMENT_PRO_RATA"])
        pkg["clauses"].append(STANDARD_CLAUSES["ULTIMATE_NET_LOSS"])
        pkg["warranties"].append(STANDARD_WARRANTIES["AGGREGATE_LIMIT"])

    # Spécifiques branche
    bc = business_class.lower()
    if 'fire' in bc or 'property' in bc:
        pkg["exclusions"].append(STANDARD_EXCLUSIONS["POLLUTION"])
        pkg["exclusions"].append(STANDARD_EXCLUSIONS["ASBESTOS"])
        pkg["warranties"].append(STANDARD_WARRANTIES["PROTECTION_FIRE"])
        pkg["warranties"].append(STANDARD_WARRANTIES["PML_ASSESSMENT"])
    if 'engineering' in bc or 'car' in bc:
        pkg["warranties"].append(STANDARD_WARRANTIES["PML_ASSESSMENT"])

    pkg["warranties"].append(STANDARD_WARRANTIES["ORIGINAL_RATES"])
    return pkg


def list_all_clauses() -> dict:
    """Retourne toute la bibliothèque (pour Streamlit picker)."""
    return {
        "clauses": STANDARD_CLAUSES,
        "exclusions": STANDARD_EXCLUSIONS,
        "warranties": STANDARD_WARRANTIES,
    }
