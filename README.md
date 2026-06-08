
# Streamlit MVP - AI Churn Early Warning Copilot

## Required files
Place these files in the same folder:
- `streamlit_app.py`
- `synthetic_saas_churn_dataset.csv`
- `best_churn_model.pkl`

## Run locally
```bash
pip install streamlit pandas numpy scikit-learn joblib
streamlit run streamlit_app.py
```

## What the MVP includes
- Portfolio dashboard
- Filterable account table
- Account detail view
- Predicted churn probability
- Risk level
- Human-readable explanation
- Suggested next action

## Run locally

```bash
git clone https://github.com/javierpinillagisbert/ai_churn_copilot.git
cd ai_churn_copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py