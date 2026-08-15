# -*- coding: utf-8 -*-
"""
CoviRisk - Site web (Streamlit)
Aide à la décision clinique : estimation de la probabilité de mortalité
hospitalière chez les patients COVID-19, via un réseau bayésien hiérarchique.

Déploiement permanent gratuit : Streamlit Community Cloud (voir README.md).
"""
import streamlit as st
from covirisk_core import BayesNet, QUESTIONS, CATEGORY_LABELS, RISK_LABELS, RISK_COLORS

st.set_page_config(
    page_title="CoviRisk - Aide à la décision clinique",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
    <style>
    .risk-box {padding: 22px; border-radius: 14px; text-align: center; margin: 14px 0;}
    .risk-box h2 {margin: 0; color: white;}
    .risk-box p {margin: 4px 0 0 0; color: white; font-size: 15px;}
    .rec-item {padding: 8px 12px; border-left: 4px solid #1a73e8; background:#f0f4ff;
               margin: 6px 0; border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 CoviRisk")
st.caption(
    "Outil d'aide à la décision clinique — estimation de la probabilité de "
    "mortalité hospitalière chez les patients COVID-19, à partir d'un réseau "
    "bayésien hiérarchique. Ne remplace pas le jugement clinique."
)

with st.expander("À propos de CoviRisk"):
    st.write(
        "CoviRisk combine 4 catégories de paramètres patient (démographie, "
        "signes vitaux, biologie, imagerie thoracique) à travers un réseau de "
        "dépendances probabilistes hiérarchiques, jusqu'à l'estimation finale "
        "de mortalité. Destiné aux professionnels de santé formés, en "
        "particulier en contexte de ressources limitées."
    )

st.divider()

obs = {}
categories = {}
for key, label, options, default, cat in QUESTIONS:
    categories.setdefault(cat, []).append((key, label, options, default))

for cat, items in categories.items():
    st.subheader(CATEGORY_LABELS[cat])
    cols = st.columns(2)
    for i, (key, label, options, default) in enumerate(items):
        with cols[i % 2]:
            labels = [o[1] for o in options]
            values = [o[0] for o in options]
            default_idx = values.index(default)
            choice = st.radio(label, labels, index=default_idx, key=f"q_{key}")
            obs[key] = values[labels.index(choice)]
    st.write("")

st.divider()

if st.button("Calculer le risque", type="primary", use_container_width=True):
    result = BayesNet().infer(obs)
    pdo = result["pdo"]
    risk_lvl = result["risk_lvl"]

    st.markdown(
        f"""
        <div class="risk-box" style="background:{RISK_COLORS[risk_lvl]};">
            <h2>{RISK_LABELS[risk_lvl]}</h2>
            <p>Probabilité estimée de mortalité : {pdo*100:.1f}%</p>
            <p style="font-size:12px;opacity:0.9;">Seuil de décision clinique p* = 0.40 · qSOFA calculé = {result['qsofa_score']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Probabilité de survie", f"{result['p_outcome'].get('fav', 0)*100:.1f}%")
    with c2:
        st.metric("Probabilité de décès", f"{result['p_outcome'].get('def', 0)*100:.1f}%")

    st.subheader("Distribution de la gravité")
    sev = result["p_severity"]
    st.bar_chart(
        {"Probabilité": [sev.get("mild", 0), sev.get("mod", 0), sev.get("sev", 0)]},
        x_label=None,
    )
    st.caption("Légère · Modérée · Sévère (de gauche à droite)")

    with st.expander("Probabilités bayésiennes a posteriori (détail des nœuds)"):
        st.write(f"- Détresse respiratoire : {result['p_distress'].get('present',0)*100:.1f}%")
        st.write(f"- Inflammation sévère : {result['p_inflam'].get('severe',0)*100:.1f}%")
        st.write(f"- Dysfonction cardiaque : {result['p_cardiac'].get('present',0)*100:.1f}%")
        st.write(f"- Dysfonction rénale : {result['p_renal'].get('present',0)*100:.1f}%")
        st.write(f"- Complications : {result['p_compl'].get('present',0)*100:.1f}%")

    st.subheader("Recommandations cliniques")
    for r in result["recs"]:
        st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)

    st.info("Outil d'aide à la décision. Ne remplace pas le jugement clinique.")

st.divider()
st.caption("CoviRisk · Réseau bayésien hiérarchique · Interface web")
