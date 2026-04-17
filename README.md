# Corporate Bankruptcy Prediction with Machine Learning

> Master's Thesis (TFM) — MSc in Data Science, Big Data & Business Analytics, **Universidad Complutense de Madrid** (February 2026)

A machine learning pipeline that predicts corporate insolvency from 18 financial indicators, benchmarking **XGBoost**, **Logistic Regression** and **Linear Regression** on a dataset of **78,682 US companies (1999–2018)**.

**🇪🇸 Versión en español:** [README.es.md](./README.es.md)

---

## 📊 Key results

| Model | ROC-AUC (Test) | Accuracy | Recall (Bankruptcy) |
|---|---|---|---|
| Linear Regression | 0.6293 | 0.0663 | 1.000 |
| Logistic Regression | 0.6679 | 0.3724 | 0.831 |
| **XGBoost (winner)** | **0.8125** | **0.8201** | 0.6025 |

- **ROC-AUC of 0.8125** on 15,737 held-out companies — competitive with sophisticated AutoML baselines.
- Handles a **1:14 class imbalance** (only 6.6% of companies bankrupt) using `scale_pos_weight`.
- Top predictors of bankruptcy: **operating risk, net profit margin, quick ratio, interest coverage, financial flexibility**.

---

## 🧠 What the project does

1. **EDA** of 78,682 company-year records with 18 normalized financial indicators across five categories:
   - *Liquidity*: quick ratio, current ratio, cash flow ratio
   - *Profitability*: ROA, ROE, net profit margin
   - *Leverage*: debt ratio, debt-to-equity, interest coverage
   - *Efficiency*: asset turnover, revenue growth, working capital ratio
   - *Risk*: management risk, industrial risk, operating risk
2. **Preprocessing**: median imputation, 80/20 stratified split, `StandardScaler` fitted on train only.
3. **Three models trained side by side** — each justified by a different role (benchmark, interpretable baseline, main model).
4. **Evaluation** focused on ROC-AUC (appropriate for imbalanced classification) plus confusion matrices and a cost analysis (a false negative is far more expensive than a false positive in credit risk).
5. **Interpretability** via feature importance and SHAP values.
6. **Productionization demo**: two synthetic company profiles (healthy vs. at-risk) to show the model behaves sensibly at the extremes.

---

## 🏗️ Repository structure

```
.
├── notebooks/
│   └── Corporate_Bankruptcy_Prediction_XGBoost.ipynb   # End-to-end analysis
├── docs/
│   └── TFM_Final.pdf                                   # Full written thesis (Spanish)
├── requirements.txt                                    # Python dependencies
├── .gitignore
├── LICENSE
├── README.md                                           # This file (English)
└── README.es.md                                        # Spanish version
```

---

## 🚀 How to run it

### Prerequisites
- Python 3.10+
- `pip` or `conda`

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/zddm5/TFM---Financials-US-Companies-.git
cd TFM---Financials-US-Companies-

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/Corporate_Bankruptcy_Prediction_XGBoost.ipynb
```

---

## 🔬 Methodology highlights

### Why XGBoost as the main model
- Captures non-linear interactions typical of financial ratios.
- Native handling of class imbalance via `scale_pos_weight ≈ 14.07`.
- Strong interpretability through feature importance and SHAP.
- No feature scaling required.

### XGBoost configuration
```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=14.07,
    random_state=42,
)
```

### Why ROC-AUC as the main metric
With a 1:14 class imbalance, accuracy is misleading and F1 depends on threshold choice. ROC-AUC summarizes discrimination ability across all thresholds. We also analyzed the cost of each error type: a **false negative (approving credit to a company that later fails) is far more expensive than a false positive**, so sensitivity was prioritized over specificity.

### Scenario simulation
Two synthetic company profiles were built by pushing the 18 indicators to extreme percentiles:
- **Healthy profile** (high profitability & liquidity, low debt & risk) → predicted survival probability **99.99%**.
- **At-risk profile** (opposite) → predicted survival probability **98.77%**, i.e. **1.23% bankruptcy risk** — the spread between extremes is economically meaningful for credit decisions.

---

## ⚠️ Limitations

- **Class imbalance (1:14)** caps recall around 60% at the default threshold — in practice, credit teams should rank companies by probability and act on the top-N riskiest.
- Model trained on 1999–2018 US data: retraining every 6–12 months is recommended and transfer to other markets/periods is not guaranteed.
- The model does **not** capture macroeconomic shocks, qualitative factors (management changes, reputation) or industry-specific cycles.
- Moderate overfitting (AUC train 0.958 vs. test 0.813), acceptable for this problem but worth monitoring.

---

## 🆚 Compared to alternatives

| Approach | Strength | Weakness vs. this work |
|---|---|---|
| Altman Z-Score (1968) | Simple, well-known | Linear assumptions, lower granularity |
| AutoML (auto-sklearn, H2O) | Higher top-line accuracy possible | Less control, lower interpretability, heavier compute |
| **This project (XGBoost)** | Interpretable, cost-aware, reproducible | Requires retraining over time |

---

## 💼 Practical applications

- **Financial institutions**: credit scoring, portfolio monitoring, loan-loss provisioning.
- **Investors**: risk-adjusted portfolio construction, early-warning systems.
- **Regulators**: systemic risk monitoring across industries.

---

## 📄 Thesis document

The full written thesis (Spanish) is available here: [docs/TFM_Final.pdf](./docs/TFM_Final.pdf).

---

## 👤 Author

**Diego José Zuniga López**
MSc Data Science, Big Data & Business Analytics — Universidad Complutense de Madrid
📍 Madrid, Spain
🔗 [GitHub @zddm5](https://github.com/zddm5)

---

## 📜 License

This project is released under the MIT License — see [LICENSE](./LICENSE) for details.
