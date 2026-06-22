# 🏥 Hospital Readmission Risk AI

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-blue.svg)](https://xgboost.readthedocs.io/)

A production-grade, end-to-end Machine Learning system designed to predict 30-day hospital readmissions. This project prioritizes **Clinical Trust, Mathematical Honesty, and Fairness** over raw accuracy metrics, adhering strictly to modern AI engineering guidelines.

---

## 🎯 The Goal
Hospitals operate on limited capacity. If a hospital can only afford to make 10 follow-up calls a day, those calls must go to the patients at the highest true risk of returning. 

This system ingests patient discharge data, calculates a calibrated risk score, audits itself for racial and gender bias, and delivers plain-English SHAP explanations via a clinical dashboard so nurses know exactly *why* a patient was flagged.

---

## 🏗️ Architecture & Engineering Principles

This project was built following the **"Think Before Coding"** and **"No Spidernets"** principles.

1. **Fairness Through Unawareness:** The model explicitly drops `race` and `gender` during training to prevent demographic bias, but strictly audits against these features post-training to prove no proxy-bias exists. (Threshold: Subgroup AUC must not fall >0.05 below overall baseline).
2. **Probability Calibration:** Raw XGBoost scores are meaningless to doctors. We use `CalibratedClassifierCV` (Sigmoid/Isotonic) to ensure that an "80% risk score" equals an exactly 80% real-world probability.
3. **Clinical Explainability (SHAP):** A risk score without an explanation is a liability. The API intercepts the XGBoost trees, runs `shap.TreeExplainer`, and translates the exact feature impacts into plain English for the UI.
4. **API-First Design:** The ML pipeline is totally decoupled from the UI. The model is served via a highly secure, Pydantic-validated FastAPI microservice.

---

## 🚀 How to Run the Project (Step-by-Step)

To run this entire system end-to-end on your local machine, follow these instructions. 

### Step 1: Environment Setup
Clone the repository and install the locked dependencies to guarantee reproducibility.
```bash
# Create and activate a virtual environment
python -m venv ml_env
.\ml_env\Scripts\activate  # Windows
# source ml_env/bin/activate  # Mac/Linux

# Install all dependencies
python -m pip install -r requirements.txt
```

### Step 2: Train the AI Pipeline
This step ingests the raw data, applies clinical ID mappings, trains the XGBoost pipeline, calibrates it, and runs the Fairness Audit.
```bash
# If you have Make installed:
make train

# OR run the scripts directly:
python src/data_prep.py
python src/train.py
```
*Note: You should see the Fairness Audit pass in the terminal, and two `.pkl` files generated in the `models/` directory.*

### Step 3: Start the Hospital API (Backend)
Bring the AI online. The FastAPI server will load the models into RAM and start listening for EHR requests.
```bash
# If you have Make installed:
make serve

# OR run the script directly:
python -m uvicorn api.app:app --reload
```
*Leave this terminal running! The API is now active on `http://127.0.0.1:8000`.*

### Step 4: Launch the Clinical Dashboard (Frontend)
Open a **brand new, second terminal window**, activate your environment, and launch the Streamlit UI for the Care Coordinators.
```bash
# Activate your environment again in the new terminal
.\ml_env\Scripts\activate

# If you have Make installed:
make dashboard

# OR run the script directly:
python -m streamlit run dashboard/streamlit_app.py
```
*A browser window will automatically pop open. You can now adjust hospital capacity, view ranked risk tables, and perform single-patient deep dives!*

---

## 📂 Project Structure

```text
Hospital_Readmission/
├── api/
│   └── app.py                  # FastAPI service (endpoints, Pydantic validation)
├── dashboard/
│   └── streamlit_app.py        # Clinical UI (Risk tables, plain-English translation)
├── Dataset/
│   ├── diabetic_data.csv       # Raw input data
│   └── clean_data.csv          # Output of data_prep.py
├── models/
│   ├── xgb_pipeline.pkl        # Base model (used for SHAP explanations)
│   └── calibrated_xgb_pipeline.pkl # Calibrated model (used for risk scores)
├── src/
│   ├── data_prep.py            # Phase 1: Data cleaning & ID mapping
│   ├── train.py                # Phase 2-5: Train, Calibrate, Fairness Audit
│   └── explain.py              # Phase 4: SHAP encapsulation logic
├── Notebooks/                  # Historic R&D and exploration notebooks
├── Dockerfile                  # Containerization for cloud deployment
├── Makefile                    # Single-command execution aliases
└── requirements.txt            # Locked environment dependencies
```

---

## 🔒 Security & Deployment Notes
If deploying to a production EHR environment (e.g., AWS ECS, Azure App Service):
* The API currently uses a hardcoded `X-API-Key` for demonstration. In production, connect this to AWS Secrets Manager or Azure Key Vault.
* `Dockerfile` is provided. The dataset is intentionally *not* copied into the Docker container to comply with PHI security best practices. The container is strictly for serving, not retraining.
* Use `uvicorn` with `gunicorn` workers for multi-threaded production scaling.
