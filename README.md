# Keap1-Bioactivity-Prediction-
Keap1-Bioactivity-Prediction is a machine learning project designed to predict the bioactivity of chemical compounds against the **Keap1 (Kelch-like ECH-associated protein 1)** target. Keap1 is a crucial regulator of the Nrf2 signaling pathway and a major drug target for oxidative stress-related diseases, inflammation, and cancer. 
This model is trained on curated, real-world bioactivity data retrieved from the **ChEMBL database**, bridging the gap between computational chemistry and data science to accelerate early-stage drug discovery.

---

## 🚀 Project Overview

The primary goal of this repository is to build a robust predictive pipeline that takes chemical structures as inputs and predicts their biological activity against Keap1. 

### Key Features
* **Data Retrieval & Curation:** Automated fetching and filtering of Keap1 bioactivity data (IC50/EC50 measurements) from ChEMBL.
* **Molecular Descriptor Generation:** Computing molecular fingerprints (such as Morgan Fingerprints/ECFP) and Lipinski's descriptors using **RDKit**.
* **Exploratory Data Analysis (EDA):** Chemical space analysis, pIC50 distribution mapping, and structural diversity checks.
* **Machine Learning Models:** Implementation and comparison of various algorithms (e.g., Random Forest, XGBoost, Support Vector Machines) to classify compounds or predict continuous pIC50 values.
* **Model Evaluation:** Thorough assessment using metrics like R², RMSE, precision, recall, and ROC-AUC curves.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Cheminformatics:** [RDKit](https://rdkit.org)
* **Database Access:** ChEMBL Web Services API
* **Machine Learning:** Scikit-learn, XGBoost / LightGBM
* **Data Science & Visualization:** Pandas, NumPy, Matplotlib, Seaborn



## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd Keap1-Bioactivity-Prediction-
```

### 2. Set Up a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r Requirements.txt
```
*(Note: If you encounter issues installing RDKit via pip, consider using a Conda environment).*

### 4. Run the Streamlit Application
Launch the interactive web application directly from your terminal:
```bash
streamlit run app.py
```


