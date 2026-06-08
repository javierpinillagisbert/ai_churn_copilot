#!/bin/bash
cd "/Users/javierpinilla/Documents/Project Website with ChatGPT/tfm_churn_project"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib
python -m streamlit run streamlit_app.py
