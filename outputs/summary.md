# Churn Analysis - Results Summary

- Rows (customers): **7,043**, Columns: **21**
- Blank TotalCharges fixed: **11** (all new customers, set to 0)
- Duplicate rows: **0**
- Overall churn rate: **26.5%** (1,869 of 7,043 customers left)

## SQL findings

**Churn By Contract**

| Contract       |   customers |   churned |   churn_rate_pct |
|:---------------|------------:|----------:|-----------------:|
| Month-to-month |        3875 |      1655 |             42.7 |
| One year       |        1473 |       166 |             11.3 |
| Two year       |        1695 |        48 |              2.8 |

**Churn By Internet Service**

| InternetService   |   customers |   churn_rate_pct |
|:------------------|------------:|-----------------:|
| Fiber optic       |        3096 |             41.9 |
| DSL               |        2421 |             19   |
| No                |        1526 |              7.4 |

**Churn By Payment Method**

| PaymentMethod             |   customers |   churn_rate_pct |
|:--------------------------|------------:|-----------------:|
| Electronic check          |        2365 |             45.3 |
| Mailed check              |        1612 |             19.1 |
| Bank transfer (automatic) |        1544 |             16.7 |
| Credit card (automatic)   |        1522 |             15.2 |

**Churn By Tenure Group**

| tenure_group    |   customers |   churn_rate_pct |
|:----------------|------------:|-----------------:|
| 1) 0-12 months  |        2186 |             47.4 |
| 2) 13-24 months |        1024 |             28.7 |
| 3) 25-48 months |        1594 |             20.4 |
| 4) 49+ months   |        2239 |              9.5 |

**Monthly Revenue Lost**

|   monthly_revenue_lost |   total_monthly_revenue |   pct_lost |
|-----------------------:|------------------------:|-----------:|
|                 139131 |                  456117 |       30.5 |

## Prediction models

Trained on 5,634 customers, tested on 1,409 unseen customers.

| Model               |   Accuracy |   Precision |   Recall |    F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|------:|----------:|
| Logistic Regression |      0.738 |       0.504 |    0.783 | 0.614 |     0.842 |
| Random Forest       |      0.762 |       0.535 |    0.781 | 0.635 |     0.843 |

Best model by ROC-AUC: **Random Forest**

## Top 10 churn drivers (Logistic Regression)

Odds ratio above 1 = more likely to leave. Below 1 = more likely to stay.

| feature                        |   coefficient |   odds_ratio |
|:-------------------------------|--------------:|-------------:|
| Contract_Two year              |        -1.413 |        0.244 |
| InternetService_Fiber optic    |         1.222 |        3.395 |
| tenure                         |        -1.156 |        0.315 |
| Contract_One year              |        -0.715 |        0.489 |
| MonthlyCharges                 |        -0.546 |        0.579 |
| TotalCharges                   |         0.487 |        1.627 |
| StreamingMovies_Yes            |         0.421 |        1.523 |
| PaymentMethod_Electronic check |         0.402 |        1.496 |
| StreamingTV_Yes                |         0.395 |        1.485 |
| PhoneService_Yes               |        -0.348 |        0.706 |
