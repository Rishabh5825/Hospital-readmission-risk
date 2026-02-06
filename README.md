# AI-Based Decision Support System for Hospital Readmissions

## Overview
Hospital readmissions within 30 days of discharge increase the burden on both patients and hospitals. Many of these readmissions are potentially preventable if high-risk patients are identified early and provided with targeted follow-up care.

This project focuses on building a machine learning–based **decision support system** that estimates readmission risk at the time of discharge, with the goal of supporting better care prioritization under limited hospital resources.

---

## Problem Statement
Hospitals often rely on reactive processes to identify readmission risk, typically after a patient has already been readmitted. There is a lack of simple, data-driven systems that proactively identify high-risk patients at discharge time.

This project explores how historical hospital data can be used to estimate 30-day readmission risk under realistic constraints such as missing data, repeated patient encounters, and class imbalance.

---

## Objective
At the time of hospital discharge, predict the probability that a patient will experience an unplanned readmission within 30 days, using only information available at discharge time.

---

## Dataset
This project uses the **Diabetes 130-US Hospitals (1999–2008)** dataset, a publicly available hospital readmission dataset.

Each row represents a single hospital encounter. Patients may appear multiple times across different encounters. The dataset includes demographic, clinical, and hospital stay–related features, along with a readmission outcome label.

The dataset is used as a **proxy** to simulate real-world hospital operational challenges and decision-making scenarios.

---

## Project Status

**Phase 1: Data exploration and problem framing — completed**  
**Phase 2: Data preprocessing and feature scope locking — completed**  
**Phase 3: Baseline leakage-free model development — completed**  
**Phase 4: Decision-centric evaluation (threshold & capacity analysis) — completed**

Work completed so far:
- Exploratory analysis of hospital readmission data
- Careful preprocessing with semantic missing-value handling
- Leakage-free baseline model using patient-level splits
- Decision-centric evaluation using threshold optimization and capacity-constrained simulations

---

## Project Structure

hospital-readmission-risk/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_decision_analysis.ipynb
└── README.md
---

## Next Steps
- Compare class-weighted training against decision-threshold optimization
- Trying tabular models (e.g., tree-based models) alongside logistic regression to assess performance, interpretability, and stability trade-offs.
- Analyze probability calibration and performance consistency across different patient subsets to improve trustworthiness for real-world deployment
