# Credit Card Fraud Detection

End-to-end fraud classification using the ULB credit-card transaction benchmark, Random Forest and XGBoost.

## Dataset
Download `creditcard.csv` from [Kaggle – ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`. The benchmark contains 284,807 transactions and 492 fraud cases. The dataset is not redistributed here; follow its source terms.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Pipeline
- Validates the real CSV schema and target.
- Uses stratified train/test split and class weighting.
- Compares Random Forest and XGBoost.
- Reports precision, recall, F1, ROC-AUC, average precision and confusion matrix.
- Saves the best pipeline and plots under `models/` and `reports/`.

**Important:** fraud data are highly imbalanced. Accuracy alone is misleading; inspect recall, precision and average precision. The saved model is a research/demo artifact, not a production payment authorization system.
