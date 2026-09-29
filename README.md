# Customer Churn Analysis (Python + SQL)

Which telecom customers are leaving, why are they leaving, and can we predict who will leave next?

**Tools:** Python, SQL (SQLite), Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
**Dataset:** [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (IBM sample dataset, 7,043 customers, 21 columns)

## Key findings

| Question | Answer |
|---|---|
| Overall churn rate | **26.5%** (1,869 of 7,043 customers left) |
| Which contract type churns most? | Month-to-month: **42.7%**, one-year: 11.3%, two-year: **2.8%** |
| Does staying longer help? | Customers in their first 12 months churn at **47.4%**; those with 49+ months churn at **9.5%** |
| Which internet service is riskiest? | Fibre optic: **41.9%** vs DSL: 19.0% |
| Which payment method is riskiest? | Electronic check: **45.3%** vs 15.2% to 19.1% for other methods |
| Revenue impact | Churned customers accounted for **30.5%** of total monthly charges |

## Model results (tested on 1,409 unseen customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 |
| Random Forest | 0.774 | 0.556 | 0.749 | 0.638 | 0.844 |

Recall is the number that matters most here: it shows how many of the customers who really left the model managed to catch (about 75% to 78%).

## Business recommendations
1. Offer discounts or perks to move month-to-month customers onto 1-year or 2-year contracts.
2. Build a special onboarding programme for the first 12 months, when customers are most likely to leave.
3. Investigate why fibre-optic customers churn so much more (price, reliability, or service quality).
4. Encourage automatic payment methods over electronic checks.

## What is in this folder
| File / folder | What it is |
|---|---|
| `run_analysis.py` | The full analysis in one script. Run this. |
| `churn_analysis.ipynb` | The same analysis as a Jupyter notebook with charts shown inline |
| `data/telco_churn.csv` | The dataset |
| `outputs/` | Charts (`charts/`), SQL result tables, model scores, and `summary.md` |
| `requirements.txt` | The Python libraries needed |

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```
Results appear in the `outputs` folder.

## Method, in short
1. **Clean:** 11 blank `TotalCharges` values (brand-new customers with zero tenure) set to 0. No duplicates found.
2. **SQL:** loaded the data into SQLite and answered business questions with `GROUP BY` and `CASE WHEN` queries.
3. **Explore:** bar charts and box plots of churn against contract, tenure, payment method, charges and internet service.
4. **Model:** 80/20 stratified train/test split, scaling and one-hot encoding in a Scikit-learn pipeline, class weights to handle the imbalance between churned and retained customers.
5. **Explain:** logistic regression coefficients converted to odds ratios to show what drives churn.

## Limitations
- This is a public sample dataset, so findings describe this data and not any real company.
- `MonthlyCharges` and `TotalCharges` are strongly related to `tenure`, which can make individual coefficients in the logistic regression look odd (for example, higher monthly charges appearing to lower churn). The odds ratios are best read as a group, not one by one.
- Precision is around 50% to 56%, meaning about half of the customers flagged as likely to leave would actually have stayed. A real team would tune the threshold based on the cost of a retention offer.
