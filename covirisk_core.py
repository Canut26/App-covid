# -*- coding: utf-8 -*-
"""
CoviRisk - Moteur de calcul (réseau bayésien hiérarchique)
Extrait à l'identique de l'application Kivy (main.py), sans dépendance graphique,
pour être partagé entre l'APK Android et le site web.
"""


class BayesNet:
    PRIORS = {
        "age":     {"low": 0.65, "high": 0.35},
        "sex":     {"m": 0.58, "f": 0.42},
        "comorb":  {"absent": 0.62, "present": 0.38},
        "bmi":     {"normal": 0.55, "high": 0.45},
        "spo2":    {"normal": 0.73, "low": 0.27},
        "rr":      {"normal": 0.68, "high": 0.32},
        "temp":    {"normal": 0.58, "high": 0.42},
        "pas":     {"normal": 0.78, "low": 0.22},
        "hr":      {"normal": 0.65, "high": 0.35},
        "crp":     {"normal": 0.42, "high": 0.58},
        "lymph":   {"normal": 0.48, "low": 0.52},
        "ddim":    {"normal": 0.35, "high": 0.65},
        "creat":   {"normal": 0.72, "high": 0.28},
        "lunginv": {"uni": 0.35, "bi": 0.65},
        "ext":     {"low": 0.58, "high": 0.42},
        "lesion":  {"gg": 0.68, "cons": 0.32},
    }

    CPT_SPO2 = {
        "absent":  {"normal": 0.540, "low": 0.460},
        "present": {"normal": 0.324, "low": 0.676},
    }
    CPT_QSOFA = {
        ("normal", "normal"): {"low": 0.512, "med": 0.421, "high": 0.067},
        ("normal", "high"): {"low": 0.312, "med": 0.578, "high": 0.110},
        ("low", "normal"): {"low": 0.285, "med": 0.598, "high": 0.117},
        ("low", "high"): {"low": 0.108, "med": 0.647, "high": 0.245},
    }
    CPT_DISTRESS = {
        ("normal", "normal", "low"): {"absent": 0.938, "present": 0.062},
        ("normal", "normal", "high"): {"absent": 0.824, "present": 0.176},
        ("normal", "high", "low"): {"absent": 0.792, "present": 0.208},
        ("normal", "high", "high"): {"absent": 0.623, "present": 0.377},
        ("low", "normal", "low"): {"absent": 0.751, "present": 0.249},
        ("low", "normal", "high"): {"absent": 0.548, "present": 0.452},
        ("low", "high", "low"): {"absent": 0.482, "present": 0.518},
        ("low", "high", "high"): {"absent": 0.218, "present": 0.782},
    }
    CPT_INFLAM = {
        ("normal", "normal"): {"weak": 0.678, "moderate": 0.256, "severe": 0.066},
        ("normal", "low"): {"weak": 0.384, "moderate": 0.428, "severe": 0.188},
        ("high", "normal"): {"weak": 0.312, "moderate": 0.456, "severe": 0.232},
        ("high", "low"): {"weak": 0.112, "moderate": 0.348, "severe": 0.540},
    }
    CPT_SEVERITY = {
        ("low", "absent", "weak"): {"mild": 0.852, "mod": 0.121, "sev": 0.027},
        ("low", "absent", "moderate"): {"mild": 0.718, "mod": 0.231, "sev": 0.051},
        ("low", "absent", "severe"): {"mild": 0.582, "mod": 0.319, "sev": 0.099},
        ("low", "present", "weak"): {"mild": 0.648, "mod": 0.283, "sev": 0.069},
        ("low", "present", "moderate"): {"mild": 0.484, "mod": 0.381, "sev": 0.135},
        ("low", "present", "severe"): {"mild": 0.318, "mod": 0.432, "sev": 0.250},
        ("med", "absent", "weak"): {"mild": 0.421, "mod": 0.452, "sev": 0.127},
        ("med", "absent", "moderate"): {"mild": 0.282, "mod": 0.518, "sev": 0.200},
        ("med", "absent", "severe"): {"mild": 0.178, "mod": 0.541, "sev": 0.281},
        ("med", "present", "weak"): {"mild": 0.248, "mod": 0.549, "sev": 0.203},
        ("med", "present", "moderate"): {"mild": 0.148, "mod": 0.582, "sev": 0.270},
        ("med", "present", "severe"): {"mild": 0.078, "mod": 0.521, "sev": 0.401},
        ("high", "absent", "weak"): {"mild": 0.152, "mod": 0.598, "sev": 0.250},
        ("high", "absent", "moderate"): {"mild": 0.082, "mod": 0.618, "sev": 0.300},
        ("high", "absent", "severe"): {"mild": 0.048, "mod": 0.551, "sev": 0.401},
        ("high", "present", "weak"): {"mild": 0.078, "mod": 0.572, "sev": 0.350},
        ("high", "present", "moderate"): {"mild": 0.031, "mod": 0.519, "sev": 0.450},
        ("high", "present", "severe"): {"mild": 0.018, "mod": 0.432, "sev": 0.550},
    }
    CPT_DEATH = {
        ("mild", "fav"): {"no": 0.978, "yes": 0.022},
        ("mild", "def"): {"no": 0.901, "yes": 0.099},
        ("mod", "fav"): {"no": 0.948, "yes": 0.052},
        ("mod", "def"): {"no": 0.781, "yes": 0.219},
        ("sev", "fav"): {"no": 0.882, "yes": 0.118},
        ("sev", "def"): {"no": 0.448, "yes": 0.552},
    }
    CPT_OUTCOME = {
        ("mild", "absent", "absent"): {"fav": 0.958, "def": 0.042},
        ("mild", "absent", "present"): {"fav": 0.872, "def": 0.128},
        ("mild", "present", "absent"): {"fav": 0.814, "def": 0.186},
        ("mild", "present", "present"): {"fav": 0.712, "def": 0.288},
        ("mod", "absent", "absent"): {"fav": 0.878, "def": 0.122},
        ("mod", "absent", "present"): {"fav": 0.751, "def": 0.249},
        ("mod", "present", "absent"): {"fav": 0.682, "def": 0.318},
        ("mod", "present", "present"): {"fav": 0.548, "def": 0.452},
        ("sev", "absent", "absent"): {"fav": 0.748, "def": 0.252},
        ("sev", "absent", "present"): {"fav": 0.581, "def": 0.419},
        ("sev", "present", "absent"): {"fav": 0.512, "def": 0.488},
        ("sev", "present", "present"): {"fav": 0.278, "def": 0.722},
    }
    CPT_CARDIAC = {
        "normal": {"absent": 0.90, "present": 0.10},
        "high": {"absent": 0.60, "present": 0.40},
    }
    CPT_RENAL = {
        ("normal", "low"): {"absent": 0.94, "present": 0.06},
        ("normal", "high"): {"absent": 0.85, "present": 0.15},
        ("high", "low"): {"absent": 0.55, "present": 0.45},
        ("high", "high"): {"absent": 0.35, "present": 0.65},
    }
    CPT_COMPL = {
        ("weak", "absent", "absent"): {"absent": 0.95, "present": 0.05},
        ("weak", "absent", "present"): {"absent": 0.78, "present": 0.22},
        ("weak", "present", "absent"): {"absent": 0.72, "present": 0.28},
        ("weak", "present", "present"): {"absent": 0.55, "present": 0.45},
        ("moderate", "absent", "absent"): {"absent": 0.82, "present": 0.18},
        ("moderate", "absent", "present"): {"absent": 0.62, "present": 0.38},
        ("moderate", "present", "absent"): {"absent": 0.55, "present": 0.45},
        ("moderate", "present", "present"): {"absent": 0.35, "present": 0.65},
        ("severe", "absent", "absent"): {"absent": 0.62, "present": 0.38},
        ("severe", "absent", "present"): {"absent": 0.40, "present": 0.60},
        ("severe", "present", "absent"): {"absent": 0.30, "present": 0.70},
        ("severe", "present", "present"): {"absent": 0.15, "present": 0.85},
    }

    @staticmethod
    def compute_qsofa_obs(rr, pas, hr):
        score = 0
        if rr == "high":
            score += 1
        if pas == "low":
            score += 1
        return score

    def infer(self, obs):
        age = obs.get("age", "low")
        sex = obs.get("sex", "f")
        comorb = obs.get("comorb", "absent")
        bmi = obs.get("bmi", "normal")
        spo2 = obs.get("spo2", "normal")
        rr = obs.get("rr", "normal")
        temp = obs.get("temp", "normal")
        pas = obs.get("pas", "normal")
        hr = obs.get("hr", "normal")
        crp = obs.get("crp", "normal")
        lymph = obs.get("lymph", "normal")
        ddim = obs.get("ddim", "normal")
        creat = obs.get("creat", "normal")
        lunginv = obs.get("lunginv", "uni")
        ext = obs.get("ext", "low")
        lesion = obs.get("lesion", "gg")

        qsofa_obs = self.compute_qsofa_obs(rr, pas, hr)

        p_qsofa_cpt = self.CPT_QSOFA[(spo2, rr)]
        p_qsofa = dict(p_qsofa_cpt)
        if pas == "low":
            shift = 0.30 * p_qsofa["low"] + 0.20 * p_qsofa["med"]
            p_qsofa["low"] -= 0.30 * p_qsofa_cpt["low"]
            p_qsofa["med"] -= 0.20 * p_qsofa_cpt["med"]
            p_qsofa["high"] += shift
        if hr == "high":
            shift = 0.10 * p_qsofa["low"]
            p_qsofa["low"] -= shift
            p_qsofa["med"] += shift * 0.6
            p_qsofa["high"] += shift * 0.4
        s = sum(p_qsofa.values())
        if s > 0:
            p_qsofa = {k: max(0, v) / s for k, v in p_qsofa.items()}
            s = sum(p_qsofa.values())
            if s > 0:
                p_qsofa = {k: v / s for k, v in p_qsofa.items()}

        p_distress_cpt = self.CPT_DISTRESS[(spo2, rr, ext)]
        p_distress = dict(p_distress_cpt)
        boost = 0.0
        if lunginv == "bi":
            boost += 0.08
        if lesion == "cons":
            boost += 0.05
        if boost > 0:
            shift = boost * p_distress["absent"]
            p_distress["absent"] -= shift
            p_distress["present"] += shift
        s = sum(p_distress.values())
        if s > 0:
            p_distress = {k: max(0, v) / s for k, v in p_distress.items()}

        p_inflam_cpt = self.CPT_INFLAM[(crp, lymph)]
        p_inflam = dict(p_inflam_cpt)
        if temp == "high":
            shift_w = 0.10 * p_inflam["weak"]
            shift_m = 0.05 * p_inflam["moderate"]
            p_inflam["weak"] -= shift_w
            p_inflam["moderate"] += shift_w * 0.4
            p_inflam["moderate"] -= shift_m
            p_inflam["severe"] += shift_w * 0.6 + shift_m
        s = sum(p_inflam.values())
        if s > 0:
            p_inflam = {k: max(0, v) / s for k, v in p_inflam.items()}

        p_cardiac = dict(self.CPT_CARDIAC[ddim])
        p_renal = dict(self.CPT_RENAL[(creat, age)])

        p_severity = {"mild": 0.0, "mod": 0.0, "sev": 0.0}
        for qv, pqv in p_qsofa.items():
            for dv, pdv in p_distress.items():
                for iv, piv in p_inflam.items():
                    key = (qv, dv, iv)
                    if key in self.CPT_SEVERITY:
                        w = pqv * pdv * piv
                        for sk in p_severity:
                            p_severity[sk] += w * self.CPT_SEVERITY[key][sk]
        if age == "high":
            shift = 0.05
            p_severity["mild"] = max(0.0, p_severity["mild"] - shift)
            p_severity["sev"] = min(1.0, p_severity["sev"] + shift)
        if bmi == "high":
            shift = 0.03
            p_severity["mild"] = max(0.0, p_severity["mild"] - shift)
            p_severity["sev"] = min(1.0, p_severity["sev"] + shift)
        if sex == "m":
            shift = 0.02
            p_severity["mild"] = max(0.0, p_severity["mild"] - shift)
            p_severity["sev"] = min(1.0, p_severity["sev"] + shift)
        s = sum(p_severity.values())
        if s > 0:
            p_severity = {k: v / s for k, v in p_severity.items()}

        p_compl = {"absent": 0.0, "present": 0.0}
        for iv, piv in p_inflam.items():
            for cv, pcv in p_cardiac.items():
                for rv, prv in p_renal.items():
                    key = (iv, cv, rv)
                    if key in self.CPT_COMPL:
                        w = piv * pcv * prv
                        for ck in p_compl:
                            p_compl[ck] += w * self.CPT_COMPL[key][ck]
        s = sum(p_compl.values())
        if s > 0:
            p_compl = {k: v / s for k, v in p_compl.items()}

        p_outcome = {"fav": 0.0, "def": 0.0}
        for sv, psv in p_severity.items():
            for cv, pcv in p_compl.items():
                for cardv, pcardv in p_cardiac.items():
                    key = (sv, cv, cardv)
                    if key in self.CPT_OUTCOME:
                        w = psv * pcv * pcardv
                        for ok in p_outcome:
                            p_outcome[ok] += w * self.CPT_OUTCOME[key][ok]
        s = sum(p_outcome.values())
        if s > 0:
            p_outcome = {k: v / s for k, v in p_outcome.items()}

        p_death = {"no": 0.0, "yes": 0.0}
        for sv, psv in p_severity.items():
            for ov, pov in p_outcome.items():
                key = (sv, ov)
                if key in self.CPT_DEATH:
                    for dk in p_death:
                        p_death[dk] += psv * pov * self.CPT_DEATH[key][dk]
        s = sum(p_death.values())
        if s > 0:
            p_death = {k: v / s for k, v in p_death.items()}

        pdo = p_death["yes"]
        if pdo >= 0.40:
            risk_lvl = "r_critical"
        elif pdo >= 0.25:
            risk_lvl = "r_high"
        elif pdo >= 0.10:
            risk_lvl = "r_moderate"
        else:
            risk_lvl = "r_low"

        if pdo >= 0.40:
            recs = list(RECS["rec_critical"])
        elif pdo >= 0.25:
            recs = list(RECS["rec_high"])
        elif pdo >= 0.10:
            recs = list(RECS["rec_mod"])
        else:
            recs = list(RECS["rec_low"])

        if p_distress.get("present", 0) > 0.5:
            recs.append(RECS["rec_distress"])
        if p_inflam.get("severe", 0) > 0.4:
            recs.append(RECS["rec_inflam"])
        if p_cardiac.get("present", 0) > 0.3:
            recs.append(RECS["rec_card"])
        if p_renal.get("present", 0) > 0.3:
            recs.append(RECS["rec_renal"])
        if p_severity.get("sev", 0) > 0.30:
            recs.append(RECS["rec_severe"])

        return {
            "p_qsofa": p_qsofa, "p_distress": p_distress,
            "p_inflam": p_inflam, "p_cardiac": p_cardiac,
            "p_renal": p_renal, "p_severity": p_severity,
            "p_compl": p_compl, "p_outcome": p_outcome,
            "p_death": p_death,
            "qsofa_score": qsofa_obs,
            "risk_lvl": risk_lvl,
            "recs": recs, "pdo": pdo,
        }


# Textes de recommandations (français), identiques à l'application mobile
RECS = {
    "rec_critical": [
        "ALERTE CRITIQUE - Soins intensifs immédiats",
        "Monitorage continu : SpO2, FC, FR, PAM",
        "O2 haut débit / VNI / ventilation mécanique",
        "Réévaluation toutes les 2 h, transfert en réanimation",
    ],
    "rec_high": [
        "RISQUE ÉLEVÉ - Hospitalisation supervisée",
        "Bilan complet : NFS, CRP, D-dimères, troponine",
        "Scanner thoracique si non réalisé",
        "Réévaluation toutes les 6 h, surveillance rapprochée",
    ],
    "rec_mod": [
        "RISQUE MODÉRÉ - Hospitalisation standard",
        "Bilan biologique et imagerie initiaux",
        "Réévaluation clinique toutes les 8 h",
    ],
    "rec_low": [
        "RISQUE FAIBLE - Surveillance standard",
        "Suivi ambulatoire si stabilité clinique",
        "Réévaluation si dégradation clinique",
    ],
    "rec_distress": "Forte probabilité de détresse respiratoire - intubation à envisager",
    "rec_inflam": "Inflammation sévère - anticoagulation prophylactique à envisager",
    "rec_card": "Dysfonction cardiaque probable - échocardiographie urgente",
    "rec_renal": "Dysfonction rénale probable - surveiller la fonction rénale",
    "rec_severe": "Profil sévère - surveillance clinique rapprochée",
}

RISK_LABELS = {
    "r_critical": "RISQUE CRITIQUE",
    "r_high": "RISQUE ÉLEVÉ",
    "r_moderate": "RISQUE MODÉRÉ",
    "r_low": "RISQUE FAIBLE",
}

RISK_COLORS = {
    "r_critical": "#d62e3d",
    "r_high": "#f28c05",
    "r_moderate": "#f39902",
    "r_low": "#21a651",
}

# Questions du formulaire : (clé, label FR, [(valeur, libellé FR), ...], valeur par défaut, catégorie)
QUESTIONS = [
    ("age", "Âge", [("low", "< 65 ans"), ("high", ">= 65 ans")], "low", "demo"),
    ("sex", "Sexe", [("f", "Féminin"), ("m", "Masculin")], "f", "demo"),
    ("comorb", "Comorbidités connues", [("absent", "Absentes"), ("present", "Présentes")], "absent", "demo"),
    ("bmi", "Indice de masse corporelle", [("normal", "Normal (< 30 kg/m²)"), ("high", "Élevé (>= 30 kg/m²)")], "normal", "demo"),

    ("spo2", "Saturation en oxygène (SpO2)", [("normal", ">= 90%"), ("low", "< 90%")], "normal", "vitals"),
    ("rr", "Fréquence respiratoire", [("normal", "<= 24 / min"), ("high", "> 24 / min")], "normal", "vitals"),
    ("temp", "Température", [("normal", "< 38,5 °C"), ("high", ">= 38,5 °C")], "normal", "vitals"),
    ("pas", "Pression artérielle", [("normal", "Normale"), ("low", "Hypotension")], "normal", "vitals"),
    ("hr", "Fréquence cardiaque", [("normal", "Normale"), ("high", "Élevée")], "normal", "vitals"),

    ("crp", "CRP (protéine C-réactive)", [("normal", "<= 50 mg/L"), ("high", "> 50 mg/L")], "normal", "labs"),
    ("lymph", "Lymphocytes", [("normal", ">= 1 G/L"), ("low", "< 1 G/L")], "normal", "labs"),
    ("ddim", "D-dimères", [("normal", "<= 500 ng/mL"), ("high", "> 500 ng/mL")], "normal", "labs"),
    ("creat", "Créatinine", [("normal", "<= 110 µmol/L"), ("high", "> 110 µmol/L")], "normal", "labs"),

    ("lunginv", "Atteinte pulmonaire (TDM)", [("uni", "Unilatérale"), ("bi", "Bilatérale")], "uni", "imaging"),
    ("ext", "Extension (TDM)", [("low", "< 50%"), ("high", ">= 50%")], "low", "imaging"),
    ("lesion", "Type de lésion (TDM)", [("gg", "Verre dépoli"), ("cons", "Condensation")], "gg", "imaging"),
]

CATEGORY_LABELS = {
    "demo": "1. Démographie",
    "vitals": "2. Signes vitaux à l'admission",
    "labs": "3. Résultats biologiques",
    "imaging": "4. Imagerie thoracique (TDM)",
}
