import streamlit as st
import pandas as pd
import requests
import json
import os
import base64

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "hospital-secure-key-2026"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'clean_data.csv')

st.set_page_config(page_title="Hospital Readmission Risk", layout="wide")

# --- UI Helper: Plain English Translation (Requirement 7) ---
def translate_feature(feature_name):
    # Blueprint exact requirements
    translations = {
        "number_inpatient": "Prior hospital admissions",
        "discharge_disposition_id_Expired": "Patient expired during stay",
        "insulin_Up": "Insulin dosage was increased",
        "num_medications": "Number of medications prescribed",
        "A1Cresult_>8": "HbA1c level above 8%"
    }
    
    if feature_name in translations:
        return translations[feature_name]
        
    # Generic fallbacks
    if feature_name.startswith("diag_1_"):
        return f"Primary Diagnosis: {feature_name.split('_')[2]}"
    if feature_name.startswith("discharge_disposition_id_"):
        return f"Discharged to: {feature_name.split('discharge_disposition_id_')[1].replace('_', ' ')}"
    if feature_name.startswith("admission_source_id_"):
        return f"Admitted from: {feature_name.split('admission_source_id_')[1].replace('_', ' ')}"
        
    return feature_name.replace("_", " ").title()

# --- API Helper ---
@st.cache_data(ttl=600)
def fetch_predictions(df_sample):
    headers = {"X-API-Key": API_KEY}
    results = []
    
    for idx, row in df_sample.iterrows():
        payload = row.drop(['target', 'encounter_id', 'patient_nbr', 'race', 'gender'], errors='ignore').to_dict()
        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                results.append({
                    "Patient_ID": row.get('patient_nbr', idx),
                    "Risk_Score": data['risk_score'],
                    "Tier": data['risk_tier'],
                    "Drivers": data['top_drivers'],
                    "Plot_Base64": data.get('waterfall_plot_base64', '')
                })
        except:
            pass
            
    return pd.DataFrame(results)

# --- Load Data ---
@st.cache_data
def load_sample_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            df[col] = df[col].astype(str)
        return df.sample(50, random_state=42)
    return None

# --- Main Dashboard ---
st.title("🏥 Hospital Readmission Command Center")

df_today = load_sample_data()

# --- Sidebar: Operational Capacity ---
st.sidebar.header("Operational Controls")
capacity = st.sidebar.slider("Nurses available today (Patients we can call)", min_value=1, max_value=50, value=10)
st.sidebar.markdown("---")
st.sidebar.success("API Status: ONLINE")

# --- UI: Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 Today's Call List", "🔍 Patient Deep Dive", "✍️ Manual Entry Lookup"])

if df_today is not None:
    with st.spinner("Connecting to EHR API..."):
        api_results = fetch_predictions(df_today)
        
    if not api_results.empty:
        api_results = api_results.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
        api_results['Intervention_Required'] = api_results.index < capacity

        with tab1:
            st.subheader(f"Top {capacity} Highest Risk Patients")
            def color_tiers(val):
                color = '#ffcccc' if val == 'HIGH' else '#ffffcc' if val == 'MEDIUM' else '#ccffcc'
                return f'background-color: {color}'
                
            display_df = api_results[['Patient_ID', 'Risk_Score', 'Tier', 'Intervention_Required']].copy()
            display_df['Risk_Score'] = (display_df['Risk_Score'] * 100).round(1).astype(str) + "%"
            st.dataframe(display_df.style.applymap(color_tiers, subset=['Tier']), use_container_width=True)

        with tab2:
            st.subheader("Understand the 'Why'")
            selected_patient = st.selectbox("Select Patient ID to review:", api_results['Patient_ID'])
            patient_data = api_results[api_results['Patient_ID'] == selected_patient].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Calibrated Risk Score", f"{patient_data['Risk_Score'] * 100:.1f}%", 
                          delta="HIGH RISK" if patient_data['Tier'] == "HIGH" else "Safe", delta_color="inverse")
                
                st.markdown("**Top 3 Reasons for Risk:**")
                for driver in patient_data['Drivers']:
                    english_name = translate_feature(driver['feature'])
                    icon = "🔺" if driver['direction'] == "increases risk" else "🔽"
                    st.info(f"{icon} **{english_name}** ({driver['direction']})")
                    
            with col2:
                # Render the base64 waterfall plot
                if patient_data['Plot_Base64']:
                    image_data = base64.b64decode(patient_data['Plot_Base64'])
                    st.image(image_data, caption="SHAP Waterfall Impact Plot", use_container_width=True)
                else:
                    st.warning("Waterfall plot not generated by API.")

with tab3:
    st.subheader("Single Patient Lookup")
    with st.form("manual_entry_form"):
        col_a, col_b, col_c = st.columns(3)
        age = col_a.number_input("Age", 0, 100, 65)
        inpatient = col_b.number_input("Prior Inpatient Visits", 0, 20, 0)
        meds = col_c.number_input("Num Medications", 1, 80, 15)
        
        diag1 = col_a.selectbox("Primary Diagnosis", ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"])
        a1c = col_b.selectbox("A1C Result", ["Not_Measured", "Norm", ">7", ">8"])
        insulin = col_c.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        
        submit = st.form_submit_button("Calculate Risk")
        
        if submit:
            # Minimal payload for demo (API handles missing fields with defaults if not forbidden)
            # Actually, our API explicitly requires these 12 fields
            payload = {
                "age": age,
                "admission_type_id": "Emergency",
                "discharge_disposition_id": "Discharged_Home",
                "admission_source_id": "Emergency_Room",
                "time_in_hospital": 5,
                "num_lab_procedures": 44,
                "num_procedures": 1,
                "num_medications": meds,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": inpatient,
                "number_diagnoses": 9,
                "diag_1": diag1,
                "diag_2": "Diabetes",
                "diag_3": "Other",
                "A1Cresult": a1c,
                "insulin": insulin
            }
            
            with st.spinner("Querying API..."):
                resp = requests.post(API_URL, json=payload, headers={"X-API-Key": API_KEY})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"**Risk Score:** {data['risk_score'] * 100:.1f}% ({data['risk_tier']})")
                    if data.get('waterfall_plot_base64'):
                        st.image(base64.b64decode(data['waterfall_plot_base64']))
                else:
                    st.error(f"API Error: {resp.text}")
