import pandas as pd
import os

def run_data_prep():
    print("--- Starting Phase 1: Data Preparation ---")
    
    # Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_path = os.path.join(base_dir, 'Dataset', 'diabetic_data.csv')
    clean_path = os.path.join(base_dir, 'Dataset', 'clean_data.csv')
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find raw data at {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape}")

    # 1. Drop Useless Columns
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 2. Define the Target Variable
    # Drop '>30', map '<30' -> 1, 'NO' -> 0
    df = df[df['readmitted'] != '>30']
    df['target'] = df['readmitted'].map({'<30': 1, 'NO': 0})
    df.drop(columns=['readmitted'], inplace=True)

    # 3. Handle Missing Values
    df.replace('?', 'Unknown', inplace=True)

    # 4. Explicit ID Mapping (The production-grade fix)
    id_mappings = {
        "admission_type_id": {
            1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
            5: "Not_Available", 6: "NULL", 7: "Trauma_Center", 8: "Not_Mapped"
        },
        "discharge_disposition_id": {
            1: "Discharged_Home", 2: "Transferred_Short_Term", 3: "Transferred_SNF",
            4: "Transferred_ICF", 5: "Transferred_Inpatient", 6: "Discharged_Home_Home_Health_Service",
            7: "Left_AMA", 8: "Discharged_Home_IV_Provider", 9: "Admitted_Inpatient",
            10: "Neonate_Discharged", 11: "Expired", 12: "Still_Patient",
            13: "Hospice_Home", 14: "Hospice_Medical_Facility", 15: "Transferred_Medicare_Approved",
            16: "Transferred_Psychiatric", 17: "Transferred_Outpatient", 18: "NULL",
            19: "Expired_Home", 20: "Expired_Medical_Facility", 21: "Expired_Unknown",
            22: "Transferred_Rehab", 23: "Transferred_Long_Term", 24: "Transferred_Medicaid_Nursing",
            25: "Not_Mapped", 26: "Transferred_Unknown", 30: "Transferred_Another_Type",
            27: "Discharged_Psychiatric", 28: "Discharged_Psychiatric", 29: "Discharged_Psychiatric"
        },
        "admission_source_id": {
            1: "Physician_Referral", 2: "Clinic_Referral", 3: "HMO_Referral",
            4: "Transfer_Hospital", 5: "Transfer_SNF", 6: "Transfer_Another_Facility",
            7: "Emergency_Room", 8: "Law_Enforcement", 9: "Not_Available",
            10: "Transfer_Critical_Access", 11: "Normal_Delivery", 12: "Premature_Delivery",
            13: "Sick_Baby", 14: "Extramural_Birth", 15: "Not_Available",
            17: "NULL", 20: "Not_Mapped", 21: "Unknown", 22: "Transfer_Hospital_Diff",
            25: "Transfer_Ambulatory", 26: "Transfer_Hospice"
        }
    }

    for col, mapping in id_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna("Unknown")

    # Ensure IDs are strings explicitly
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    # Save to disk
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    
    print(f"Data Prep Complete! Clean data saved to: {clean_path}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    run_data_prep()
