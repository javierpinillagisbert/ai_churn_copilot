# Semrush Retention Intelligence Copilot V3 - Synthetic Demo Package

This package contains a Semrush-like synthetic churn dataset and a Streamlit Copilot adapted for CSM storytelling.

## Files
- `synthetic_semrush_churn_dataset_v3.csv` — synthetic demo dataset (3,500 rows, 139 columns)
- `data_dictionary_semrush_churn_v3.csv` — field dictionary and model usage flag
- `model_feature_importance_semrush_v3.csv` — model feature importance export
- `generate_semrush_churn_dataset_v3.py` — reproducible dataset generator
- `train_semrush_churn_model_v3.py` — retrainable model script
- `best_churn_model_semrush_v3.pkl` — trained RandomForest model package
- `streamlit_app_semrush_v3.py` — Semrush-like Streamlit Copilot with CSM CTAs and Scenario Simulator

## Run locally
```bash
cd /path/to/project
python train_semrush_churn_model_v3.py
python -m streamlit run streamlit_app_semrush_v3.py
```

## Disclaimer
This dataset is synthetic and is inspired only by public product/KB information. It does not include real Semrush customer data, internal systems, or confidential metrics.
