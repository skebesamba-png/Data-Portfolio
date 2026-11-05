import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# CUSTOMER CHURN PREDICT
# Tableau de bord & Prédiction
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predict",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modele" / "modele_regression_logistique.pkl"
VARIABLES_PATH = BASE_DIR / "modele" / "variables_retenues.pkl"

N_CLIENTS = 6957
CHURN_RATE = 30.16
ACCURACY = 88.72
PRECISION = 81.69
RECALL = 80.71
F1_TEST = 81.20
F1_CV = 81.94

# Résultats de l'analyse exploratoire et de la sélection des variables
TAUX_NON = 69.84
TAUX_OUI = 30.16

CRAMER_DATA = pd.DataFrame({
    "Variable": [
        "Smartphone",
        "Abonnement TV",
        "Sauvegarde contacts",
        "Services messageries",
        "Abonnement musique",
        "Multi-SIM",
        "Plusieurs lignes fixe",
        "Accès 4G mobile",
        "Type internet",
        "Orange Money"
    ],
    "V de Cramer": [
        0.230409,
        0.201353,
        0.199120,
        0.192178,
        0.185406,
        0.182564,
        0.170772,
        0.163641,
        0.104837,
        0.076734
    ]
})

MI_DATA = pd.DataFrame({
    "Variable": [
        "Dépense mensuelle Crédit",
        "Ancienneté Telco",
        "Dépense mensuelle Data"
    ],
    "Information mutuelle": [
        0.148860,
        0.113224,
        0.020890
    ]
})

# -----------------------------
# VARIABLES DU MODÈLE
# -----------------------------
VARIABLES_REQUISES = [
    "anciennete_telco_mois",
    "depence_mensuelle_data",
    "depense_mensuelle_dredit",
    "smartphone",
    "abonnement_tv",
    "sauvegarde_contact_en_ligne",
    "services_messageries",
    "abonnement_musique",
    "multisim",
    "plusieurs_ligne_telephonie_fixe",
    "acces_connexion_4G_mobile",
    "type_internet",
    "utlisation_orange_money"
]

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
    }

    [data-testid="stSidebar"] * {
        color: #eef3f8 !important;
    }

    .brand {
        padding: 10px 4px 24px 4px;
    }

    .brand-title {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #ffffff;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #9ca9bc !important;
        margin-top: 5px;
    }

    .hero {
        background: linear-gradient(135deg, #0b1220 0%, #17243a 100%);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 12px 35px rgba(11,18,32,.12);
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #9fb2c9;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 8px 0 8px 0;
    }

    .hero-text {
        color: #c9d3df;
        font-size: 15px;
        max-width: 780px;
        line-height: 1.6;
    }

    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #101828;
        margin: 8px 0 15px 0;
    }

    .kpi {
        background: white;
        border: 1px solid #e7ebf1;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 7px 24px rgba(16,24,40,.05);
        min-height: 126px;
    }

    .kpi-label {
        color: #667085;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #101828;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .kpi-note {
        color: #98a2b3;
        font-size: 11px;
        margin-top: 5px;
    }

    .card {
        background: white;
        border: 1px solid #e7ebf1;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 7px 24px rgba(16,24,40,.045);
    }

    .risk-high {
        background: #fff3f0;
        border: 1px solid #ffd5cc;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
    }

    .risk-low {
        background: #f0fdf4;
        border: 1px solid #ccefd7;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
    }

    .risk-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .risk-score {
        font-size: 48px;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -2px;
    }

    .risk-text {
        color: #667085;
        font-size: 13px;
    }

    .prob-card {
        background: #ffffff;
        border: 1px solid #e7ebf1;
        border-radius: 16px;
        padding: 17px 20px;
    }

    .prob-label {
        color: #667085;
        font-size: 12px;
        font-weight: 600;
    }

    .prob-value {
        color: #101828;
        font-size: 23px;
        font-weight: 800;
        margin-top: 4px;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 11px;
        padding: 28px 0 8px 0;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 48px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CHARGEMENT DU MODÈLE
# -----------------------------
@st.cache_resource
def charger_modele():
    chemin_modele = (
        Path(__file__).resolve().parent
        / "modele_regression_logistique.pkl"
    )

    if not chemin_modele.exists():
        raise FileNotFoundError(
            f"Le fichier du modèle est introuvable : {chemin_modele}"
        )

    return joblib.load(chemin_modele)


modele = charger_modele()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-title">CUSTOMER CHURN</div>
        <div class="brand-title">PREDICT</div>
        <div class="brand-subtitle">Analyse • Scoring • Décision</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["📊 Tableau de bord", "🎯 Prédiction"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#9ca9bc;'>"
        "Présenté par M.Samba KEBE Ingénieur Statisticien & Informaticien - DATA ANALYST | MACHINE LEARNING & BUSINESS INTELLIGENCE<br>"
        "</div>",
        unsafe_allow_html=True
    )

# ============================================================
# TABLEAU DE BORD
# ============================================================
if menu == "📊 Tableau de bord":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">Plateforme d'analyse prédictive</div>
        <div class="hero-title">Customer Churn Predict</div>
        <div class="hero-text">
            Une solution de scoring client conçue pour identifier les profils
            susceptibles de se désabonner et transformer les données en
            décisions de fidélisation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Vue d'ensemble</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    kpis = [
        ("CLIENTS ANALYSÉS", f"{N_CLIENTS:,}".replace(",", " "), "Base d'étude"),
        ("TAUX DE DÉSABONNEMENT", f"{CHURN_RATE:.2f} %", "Classe « Oui »"),
        ("F1-SCORE TEST", f"{F1_TEST:.2f} %", "Modèle optimisé"),
        ("F1-SCORE CV", f"{F1_CV:.2f} %", "Validation croisée · 5 plis"),
    ]

    for col, (label, value, note) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # RAPPORT ANALYTIQUE — STYLE POWER BI
    # ========================================================

    st.markdown(
        "<div class='section-title'>Analyse du portefeuille client</div>",
        unsafe_allow_html=True
    )

    # Ligne 1 : répartition de la cible + indicateurs analytiques
    r1, r2 = st.columns([1, 1.35])

    with r1:
        st.markdown("""
        <div class="card">
            <div style="font-size:14px;font-weight:800;color:#101828;margin-bottom:8px;">
                Répartition des clients
            </div>
            <div style="font-size:12px;color:#667085;margin-bottom:16px;">
                Distribution de la variable cible
            </div>
        """, unsafe_allow_html=True)

        repartition = pd.DataFrame(
            {"Clients (%)": [TAUX_NON, TAUX_OUI]},
            index=["Non-désabonnement", "Désabonnement"]
        )
        st.bar_chart(
            repartition,
            y="Clients (%)",
            height=250
        )

        st.markdown("""
            <div style="display:flex;justify-content:space-between;
                        font-size:12px;color:#667085;margin-top:4px;">
                <span>Non-désabonnement : <b>69,84 %</b></span>
                <span>Désabonnement : <b>30,16 %</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class="card">
            <div style="font-size:14px;font-weight:800;color:#101828;">
                Indicateurs clés du portefeuille
            </div>
            <div style="font-size:12px;color:#667085;margin-top:4px;margin-bottom:18px;">
                Lecture synthétique de la population étudiée
            </div>
        """, unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.metric("Clients non désabonnés", "4 859", "69,84 %")
            st.metric("Clients désabonnés", "2 098", "30,16 %")
        with b:
            st.metric("Variables retenues", "13", "3 quantitatives • 10 qualitatives")
            st.metric("Clients analysés", "6 957", "Base finale")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 2 : variables qualitatives et quantitatives
    q1, q2 = st.columns(2)

    with q1:
        st.markdown("""
        <div class="card">
            <div style="font-size:14px;font-weight:800;color:#101828;">
                Variables qualitatives les plus associées
            </div>
            <div style="font-size:12px;color:#667085;margin-top:4px;margin-bottom:12px;">
                V de Cramer — intensité de l'association avec le désabonnement
            </div>
        """, unsafe_allow_html=True)

        cramer_plot = CRAMER_DATA.sort_values("V de Cramer", ascending=True)
        st.bar_chart(
            cramer_plot.set_index("Variable"),
            y="V de Cramer",
            height=360
        )

        st.markdown("""
            <div style="font-size:11px;color:#98a2b3;margin-top:4px;">
                Les variables sont classées selon l'intensité de leur association.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div class="card">
            <div style="font-size:14px;font-weight:800;color:#101828;">
                Variables quantitatives les plus informatives
            </div>
            <div style="font-size:12px;color:#667085;margin-top:4px;margin-bottom:12px;">
                Information mutuelle avec la variable cible
            </div>
        """, unsafe_allow_html=True)

        mi_plot = MI_DATA.sort_values("Information mutuelle", ascending=True)
        st.bar_chart(
            mi_plot.set_index("Variable"),
            y="Information mutuelle",
            height=360
        )

        st.markdown("""
            <div style="font-size:11px;color:#98a2b3;margin-top:4px;">
                Une valeur élevée traduit une information plus importante
                pour caractériser la cible.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 3 : synthèse analytique
    st.markdown(
        "<div class='section-title'>Synthèse analytique</div>",
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("""
        <div class="card">
            <div style="font-size:12px;color:#667085;">INDICATEUR 01</div>
            <div style="font-size:22px;font-weight:800;color:#101828;margin-top:5px;">
                30,16 %
            </div>
            <div style="font-size:12px;color:#667085;line-height:1.6;margin-top:6px;">
                des clients de la base sont concernés par le désabonnement.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="card">
            <div style="font-size:12px;color:#667085;">INDICATEUR 02</div>
            <div style="font-size:22px;font-weight:800;color:#101828;margin-top:5px;">
                Smartphone
            </div>
            <div style="font-size:12px;color:#667085;line-height:1.6;margin-top:6px;">
                présente la plus forte association parmi les variables qualitatives
                retenues selon le V de Cramer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="card">
            <div style="font-size:12px;color:#667085;">INDICATEUR 03</div>
            <div style="font-size:22px;font-weight:800;color:#101828;margin-top:5px;">
                Crédit
            </div>
            <div style="font-size:12px;color:#667085;line-height:1.6;margin-top:6px;">
                la dépense mensuelle Crédit possède la plus forte information
                mutuelle parmi les variables quantitatives retenues.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Lecture métier</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-size:15px;font-weight:700;color:#101828;">
            Anticiper plutôt que subir le désabonnement
        </div>
        <div style="font-size:13px;color:#667085;line-height:1.7;margin-top:8px;">
            L'analyse met en évidence une population de clients à surveiller,
            ainsi que des caractéristiques particulièrement associées au
            désabonnement. Le scoring individuel permet ensuite de transformer
            ces constats en une aide à la décision pour les actions de fidélisation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PRÉDICTION
# ============================================================
else:

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">Scoring individuel</div>
        <div class="hero-title">Prédire le risque client</div>
        <div class="hero-text">
            Renseignez les caractéristiques du client pour obtenir sa
            probabilité de désabonnement et une décision interprétable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if modele is None:
        st.error(
            "Modèle introuvable. Placez le fichier "
            "'modele_regression_logistique.pkl' dans le dossier 'modele'."
        )
        st.stop()

    st.markdown("<div class='section-title'>Profil du client</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        anciennete = st.number_input(
            "Ancienneté (mois)",
            min_value=0.0,
            max_value=300.0,
            value=24.0,
            step=1.0
        )
        smartphone = st.selectbox("Smartphone", ["Oui", "Non"])
        multisim = st.selectbox("Multi-SIM", ["Oui", "Non"])
        plusieurs_ligne = st.selectbox(
            "Plusieurs lignes téléphonie fixe", ["Oui", "Non"]
        )

    with col2:
        type_internet = st.selectbox(
            "Type d'internet", ["sobox_fibre", "sobox_fixe", "Non"]
        )
        acces_4g = st.selectbox(
            "Accès connexion 4G mobile", ["Oui", "Non", "service internet indisponible"]
        )
        messageries = st.selectbox(
            "Services de messageries", ["Oui", "Non", "service internet indisponible"]
        )
        sauvegarde = st.selectbox(
            "Sauvegarde contact en ligne", ["Oui", "Non", "service internet indisponible"]
        )

    with col3:
        musique = st.selectbox("Abonnement musique", ["Oui", "Non", "service internet indisponible"])
        tv = st.selectbox("Abonnement TV", ["Oui", "Non", "service internet indisponible"])
        orange_money = st.selectbox(
            "Utilisation Orange Money", ["Oui", "Non"]
        )

    st.markdown("<div class='section-title'>Dépenses mensuelles</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        data_spend = st.number_input(
            "Dépense mensuelle Data",
            min_value=500.0,
            value=5000.0,
            step=100.0
        )

    with col2:
        credit_spend = st.number_input(
            "Dépense mensuelle Crédit",
            min_value=1000.0,
            value=50000.0,
            step=1000.0
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 1.2, 1])
    with col_btn:
        lancer = st.button(
            "🔍 ANALYSER LE RISQUE",
            use_container_width=True,
            type="primary"
        )

    if lancer:
        donnees_client = pd.DataFrame([{
            "anciennete_telco_mois": anciennete,
            "depence_mensuelle_data": data_spend,
            "depense_mensuelle_dredit": credit_spend,
            "smartphone": smartphone,
            "abonnement_tv": tv,
            "sauvegarde_contact_en_ligne": sauvegarde,
            "services_messageries": messageries,
            "abonnement_musique": musique,
            "multisim": multisim,
            "plusieurs_ligne_telephonie_fixe": plusieurs_ligne,
            "acces_connexion_4G_mobile": acces_4g,
            "type_internet": type_internet,
            "utlisation_orange_money": orange_money
        }])

        try:
            prediction = int(modele.predict(donnees_client)[0])
            probabilites = modele.predict_proba(donnees_client)[0]

            prob_non = float(probabilites[0])
            prob_oui = float(probabilites[1])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Résultat de l'analyse</div>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div class="risk-title">⚠️ Risque élevé</div>
                    <div class="risk-score">{prob_oui * 100:.2f} %</div>
                    <div class="risk-text">
                        Probabilité estimée de désabonnement
                    </div>
                    <div style="margin-top:16px;font-size:14px;font-weight:800;">
                        DÉCISION : DÉSABONNEMENT
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.warning(
                    "Ce profil présente une probabilité élevée de désabonnement. "
                    "Une action de fidélisation peut être envisagée."
                )
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div class="risk-title">✓ Risque faible</div>
                    <div class="risk-score">{prob_non * 100:.2f} %</div>
                    <div class="risk-text">
                        Probabilité estimée de non-désabonnement
                    </div>
                    <div style="margin-top:16px;font-size:14px;font-weight:800;">
                        DÉCISION : NON-DÉSABONNEMENT
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success(
                    "Ce profil présente une probabilité plus élevée de rester client."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">PROBABILITÉ — NON-DÉSABONNEMENT</div>
                    <div class="prob-value">{prob_non * 100:.2f} %</div>
                </div>
                """, unsafe_allow_html=True)

            with p2:
                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">PROBABILITÉ — DÉSABONNEMENT</div>
                    <div class="prob-value">{prob_oui * 100:.2f} %</div>
                </div>
                """, unsafe_allow_html=True)

            resultat = donnees_client.copy()
            resultat["Probabilite_Non"] = round(prob_non * 100, 2)
            resultat["Probabilite_Oui"] = round(prob_oui * 100, 2)
            resultat["Decision"] = (
                "Désabonnement" if prediction == 1 else "Non-désabonnement"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Télécharger le résultat",
                data=resultat.to_csv(index=False).encode("utf-8-sig"),
                file_name="prediction_client.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(
                "La prédiction n'a pas pu être effectuée. "
                "Vérifiez que les modalités saisies correspondent bien "
                "à celles utilisées lors de l'entraînement du modèle."
            )
            st.exception(e)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    CUSTOMER CHURN PREDICT · Analyse prédictive du désabonnement client<br>
    @Samba KEBE · Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
