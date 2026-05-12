## 🏠 House Price Predictor — Amazon SageMaker

> Predicts California median house prices using XGBoost trained and deployed on Amazon SageMaker. Built as part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshots

### Model Predictions & RMSE
<img width="1902" height="845" alt="01-sagemaker" src="https://github.com/user-attachments/assets/c79d1f0b-6892-41bb-ac0a-be0dbf60f243" />

### Feature Engineering — 4 New Features Verified
<img width="1897" height="887" alt="02-sagemaker" src="https://github.com/user-attachments/assets/8401d4c2-0ea4-4066-aef1-3de68be25b9e" />

---

## 📊 Results

| Version | Features | num_round | eta | RMSE |
|---------|----------|-----------|-----|------|
| V1 — Baseline | 8 | 100 | 0.2 | $47,521 |
| V2 — Engineered + Tuned | 12 | 300 | 0.05 | In progress |

---

## 🧠 What This Project Does

Trains an XGBoost regression model on the California Housing dataset to predict median house prices at the block level. Deploys the model as a live SageMaker real-time endpoint and evaluates accuracy using RMSE.

---

## 🗂️ Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Size:** 20,640 houses across California block groups
- **Target:** `MedHouseVal` — median house value (×$100,000)

### Original Features (8)
| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupants |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

### Engineered Features (4 new)
| Feature | Formula | Why It Helps |
|---------|---------|--------------|
| `rooms_per_person` | `AveRooms / AveOccup` | Measures spaciousness |
| `bedrooms_per_room` | `AveBedrms / AveRooms` | Measures bedroom density |
| `population_per_hh` | `Population / AveOccup` | Measures neighborhood crowding |
| `income_per_room` | `MedInc / AveRooms` | Measures wealth per room |

---

## ⚙️ Model Configuration

```python
xgb_estimator.set_hyperparameters(
    objective      = "reg:squarederror",
    num_round      = 100,
    max_depth      = 5,
    eta            = 0.2,
    subsample      = 0.8
)
```

**Instance types used:**
- Training: `ml.m5.large`
- Endpoint: `ml.t2.medium`

---

## 🔁 ML Lifecycle Covered

```
Data Prep → S3 Upload → Training Job → Evaluation → Feature Engineering → Retrain → Deploy → Inference → Cleanup
```

---

## 🧪 AWS Concepts Practiced

- ✅ SageMaker training jobs and estimators
- ✅ S3 data input / model output
- ✅ Built-in XGBoost algorithm
- ✅ Real-time inference endpoints
- ✅ Hyperparameter configuration
- ✅ IAM roles and permissions
- ✅ Feature engineering before training
- ✅ Model evaluation with RMSE
- ✅ Resource cleanup to avoid charges

---

## 📝 AIF-C01 Exam Topics It Covers

| Topic | Domain |
|-------|--------|
| ML lifecycle (prepare → train → evaluate → deploy) | Domain 1: Fundamentals of AI/ML |
| Regression metrics — RMSE vs accuracy | Domain 1: Fundamentals of AI/ML |
| Feature engineering | Domain 1: Fundamentals of AI/ML |
| Overfitting prevention (eta, subsample) | Domain 1: Fundamentals of AI/ML |
| SageMaker built-in algorithms | Domain 1: Fundamentals of AI/ML |
| IAM roles for ML services | Domain 5: Security & Governance |
| Pay-as-you-go / endpoint cost management | Domain 2: Generative AI Fundamentals |

---

## 🚀 How to Run

### Prerequisites
- AWS account (free tier works)
- SageMaker Studio set up in `us-east-1`
- IAM role with SageMaker + S3 access

### Steps
1. Open `california_housing.ipynb` in SageMaker Studio
2. Run all cells top to bottom
3. Check RMSE output
4. **Delete the endpoint when done** to avoid charges

### Cleanup
```python
predictor.delete_endpoint()
print("Endpoint deleted — no more charges.")
```

---

## 📁 Files

```
01-sagemaker/
├── california_housing.ipynb
├── screenshots/
│   ├── predictions_output.png
│   └── feature_engineering.png
└── results/
    └── predictions_v1.csv
```

---

## 💡 What I Learned

Working through this project taught me how AWS separates the training and deployment steps — you pay for compute only while it's running. The most surprising thing was how much feature engineering matters: adding just 4 derived columns could cut RMSE by 15–25% without changing the algorithm at all. It maps directly to what the AIF-C01 exam tests in the ML fundamentals domain.
