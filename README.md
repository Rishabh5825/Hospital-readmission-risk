# AI-Based Decision Support System for Hospital Readmissions

## Overview
Hospital readmissions within 30 days of discharge increase the burden on both patients and hospitals. Many of these readmissions are preventable if high-risk patients are identified early and provided with better follow-up care.

This project focuses on building a machine learning–based decision-support system that estimates readmission risk at the time of discharge.

---

## Problem Statement
Hospitals often rely on reactive processes to identify readmission risk. There is a lack of simple, data-driven systems that help care teams identify high-risk patients before readmission occurs.

This project aims to explore how historical hospital data can be used to predict 30-day readmission risk under realistic data constraints.

---

## Objective
To build a probabilistic model that estimates the likelihood of a patient being readmitted within 30 days of discharge, using only information available at discharge time.

---

## Dataset
This project uses the **Diabetes 130-US Hospitals (1999–2008)** dataset, a publicly available hospital readmission dataset.

Each row represents a single hospital admission. Patients may appear multiple times across different encounters. The dataset includes demographic, clinical, and hospital stay–related features, along with a readmission outcome label.

---

## Project Status
🚧 **Phase 0 – Dataset understanding and problem framing**
   **Phase 1: Data understanding and problem scoping — completed** 
   **Phase 2: Feature selection and scope locking — in progress**


Work completed so far:
- Dataset exploration and familiarization
- Problem definition and scope finalization
- Initial data exploration and assumptions are documented in `notebooks/01_data_understanding.ipynb`.


---

## Project Structure
hospital-readmission-risk/
├── data/
├── notebooks/
├── docs/
└── README.md

---

## Next Steps
- Data understanding and cleaning
- Exploratory data analysis
- Baseline model development
