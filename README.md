# MOT Odometer Fraud Detection  

Detecting mileage fraud in UK MOT records using unsupervised machine learning (no labelled data required).

---

## Project Structure  

📁 project  
│  
├── fyp_final_file.ipynb       # Main notebook — run this  
├── fyp_file__final.ipynb      # Draft  
├── new_fyp_file_(1).ipynb     # Draft  
├── Dataset                   # Dataset link  
└── README.md  

---

## What is this project?  

Every car in the UK needs an MOT test each year. During that test, the mileage on the odometer is recorded. Some dishonest sellers wind back (or "clock") the odometer to make a car look like it has done fewer miles than it actually has — this inflates the car's value and hides wear and tear.  

This project uses machine learning to automatically spot suspicious vehicles in the official DVSA MOT dataset by analysing how a car's driving behaviour changes over time. If a car has suddenly "driven much less" than usual, that is a red flag.  

---

## Goal  

Compare four unsupervised anomaly detection algorithms to identify odometer fraud:

| Model | Type |
|------|------|
| Isolation Forest | Tree-based anomaly detection |
| One-Class SVM | Boundary-based detection |
| Local Outlier Factor (LOF) | Density-based detection |
| Elliptic Envelope | Statistical detection |

---

## Dataset  

- Source: UK Government DVSA MOT records (2018–2024)  
- Size: 40,000 vehicles (10% sample)  
- Key columns: vehicle_id, test_date, test_mileage, make, fuel_type, first_use_date  

Dataset is not included. Place it here:  
`MOT/mot_2018_2024_10percent_by_vehicle.csv.gz`

--- 
## Libraries Used  

- pandas — data processing  
- numpy — numerical operations  
- scikit-learn — machine learning models  
- matplotlib / seaborn — visualisation  
- matplotlib-venn — model comparison  
- joblib — saving and loading models  

---
## Requirements  

- Python 3.8+  
- Google Colab or Jupyter Notebook  
- 2GB RAM minimum for the 40,000 vehicle sample  

---

## How it Works  

### Step 1 — Feature Engineering  
Behavioural features created:

| Feature | Description |
|--------|------------|
| mileage_diff | Miles since last MOT |
| miles_per_day | Daily driving intensity |
| usage_trend_shift | Change vs previous year (fraud signal) |
| car_age_at_test | Vehicle age |

---

### Step 2 — Fraud Injection  
600 synthetic fraud cases are added to simulate mileage rollback ("Clocked Commuter").

---

### Step 3 — Data Cleaning  
- Remove vehicles >300 miles/day  
- Remove >50% usage drops  
- Remove negative mileage values  

---

### Step 4 — Model Training  
Models trained only on normal data to learn typical behaviour.

---

### Step 5 — Testing & Evaluation  
Test dataset includes:
- 600 fraud vehicles  
- 2,000 normal vehicles  
- Negative mileage cases in the dataset 

---
## Key Concepts  

- **Clocked Commuter:** A fraudulent vehicle where the odometer has been partially rolled back. Mileage increases slightly but remains far below its historical trend.  

- **Recall:** The proportion of actual fraud cases detected. A recall of 90% means 9 out of 10 fraud cases are identified.  

- **Precision:** The proportion of detected fraud cases that are truly fraudulent. High precision reduces false accusations.  

---

## Results  

| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|----------|--------|---------|
| One-Class SVM | 95.07% | 91.79% | 90.09% | 90.93% |
| Isolation Forest | 89.48% | 86.77% | 72.38% | 79.17% |
| LOF | 77.11% | 72.83% | 26.55% | 38.92% |
| Elliptic Envelope | 80.49% | 82.30% | 36.86% | 50.91% |

**Best Model:** One-Class SVM — detects 9/10 fraud cases with low false positives.

---

## How to Run  

1. Upload dataset to Google Drive  
   `My Drive/MOT/mot_2018_2024_10percent_by_vehicle.csv.gz`

2. Open notebook in Google Colab  

3. Install dependencies (if running locally):```bash pip install pandas numpy scikit-learn matplotlib seaborn matplotlib-venn joblib
---
## Author  

**Irzam Latif** 
(University of Hertfordshire)  
Final Year Project — MSc Data Science  

Dataset: DVSA MOT Data — UK Open Government Licence  


