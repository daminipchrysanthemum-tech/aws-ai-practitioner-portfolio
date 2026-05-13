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

---

# 🔍 Photo Content Moderator — Amazon Rekognition

> Automatically scans images for inappropriate or unsafe content using Amazon Rekognition's pre-trained computer vision models. Built as part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshot

### Final Moderation Report

<img width="1901" height="900" alt="Photo Content Moderator" src="https://github.com/user-attachments/assets/43983b01-479c-4228-891a-5f0a62c98643" />

---

## 📊 Results

| Image | Moderation Status |
|-------|-----------------|
| safe_park.jpg | ✅ Clean |
| test_safe.jpg | ✅ Clean |
| test_nature.jpg | ✅ Clean |
| test_warning.jpg | ✅ Clean |

All 4 images correctly identified as safe content with `MinConfidence=10`, demonstrating that the moderator works end-to-end and correctly returns Clean for non-flagged images.

---

## 🧠 What This Project Does

Uses **Amazon Rekognition's `detect_moderation_labels` API** to scan images stored in S3 and classify them as safe or flagged. For each image, Rekognition returns a list of moderation labels (e.g. Explicit Nudity, Violence, Drugs, Alcohol, Gambling) with confidence scores from 0–100. The notebook loops through a batch of images and produces a final summary report as a pandas DataFrame.

---

## 🔑 Key Difference from SageMaker

| | SageMaker | Rekognition |
|--|-----------|-------------|
| Model training | You train your own | Pre-trained by AWS |
| Endpoints | You deploy and manage | No endpoints needed |
| Cleanup | Must delete endpoint | Nothing to delete |
| Billing | Per instance-hour | Per API call (~$0.001/image) |
| Expertise needed | ML knowledge required | No ML expertise needed |

---

## ⚙️ How It Works

```
Image uploaded to S3
        ↓
rekognition.detect_moderation_labels()
        ↓
Returns list of ModerationLabels with confidence scores
        ↓
If labels empty → "Clean"
If labels found → "Flagged: <category>"
        ↓
Results compiled into pandas DataFrame report
```

---

## 🗂️ Moderation Categories Rekognition Can Detect

| Category | Examples |
|----------|---------|
| Explicit Content | Adult nudity, suggestive content |
| Violence | Graphic violence, weapon use |
| Drugs & Tobacco | Drug use, drug paraphernalia |
| Alcohol | Drinking, alcohol products |
| Gambling | Casino imagery |
| Hate Symbols | Extremist symbols |
| Rude Gestures | Offensive hand gestures |

---

## 💻 Core Code

```python
import boto3
import pandas as pd

rekognition = boto3.client("rekognition", region_name="us-east-1")

def moderate_image(bucket, image_key, min_confidence=50):
    response = rekognition.detect_moderation_labels(
        Image={
            "S3Object": {
                "Bucket": bucket,
                "Name":   image_key
            }
        },
        MinConfidence=min_confidence
    )

    labels = [l["Name"] for l in response["ModerationLabels"]]
    status = "Clean" if not labels else ", ".join(labels)
    return status

# Batch scan multiple images
all_files = ["safe_park.jpg", "test_safe.jpg", "test_nature.jpg", "test_warning.jpg"]
results_data = []

for filename in all_files:
    response = rekognition.detect_moderation_labels(
        Image={"S3Object": {"Bucket": bucket_name, "Name": f"{s3_prefix}/{filename}"}},
        MinConfidence=10
    )
    labels = [l["Name"] for l in response["ModerationLabels"]]
    status = "Clean" if not labels else ", ".join(labels)
    results_data.append({"Image Name": filename, "Moderation Status": status})

df = pd.DataFrame(results_data)
print("--- FINAL MODERATION REPORT ---")
display(df)
```

---

## 🧪 AWS Concepts Practiced

- ✅ Amazon Rekognition `detect_moderation_labels` API
- ✅ Amazon Rekognition `detect_labels` API (object detection)
- ✅ S3 image storage and API integration
- ✅ Confidence score thresholds (`MinConfidence`)
- ✅ Pre-trained vs custom models (no training needed)
- ✅ Serverless / pay-per-call pricing model
- ✅ `boto3` client for AWS AI services
- ✅ pandas DataFrame for results reporting
- ✅ Batch processing multiple images

---

## 📝 AIF-C01 Exam Topics It Covers

| Topic | Domain |
|-------|--------|
| Pre-built AI services vs custom models | Domain 1: Fundamentals of AI/ML |
| Confidence scores and thresholds | Domain 1: Fundamentals of AI/ML |
| Computer vision use cases | Domain 1: Fundamentals of AI/ML |
| Responsible AI — content moderation | Domain 4: Responsible AI |
| Serverless / pay-per-call pricing | Domain 2: Generative AI Fundamentals |
| When to use Rekognition vs SageMaker | Domain 1: Fundamentals of AI/ML |
| S3 as data source for AI services | Domain 5: Security & Governance |

---

## 🚀 How to Run

### Prerequisites
- AWS account with SageMaker Studio access
- IAM role with Rekognition + S3 permissions
- Images uploaded to your SageMaker S3 bucket

### Steps
1. Open `rekognition_content_moderator.ipynb` in SageMaker Studio
2. Run all cells top to bottom
3. View the Final Moderation Report table at the bottom

### No cleanup needed
Unlike SageMaker, Rekognition has no endpoints or instances to shut down. There are no running resources — you only pay per API call.

---

## 💡 Key Concept: MinConfidence Threshold

```python
# Strict — only report high-confidence findings (fewer false positives)
MinConfidence=80

# Balanced — good for most use cases
MinConfidence=50

# Sensitive — catch everything, more false positives
MinConfidence=10
```

Adjusting `MinConfidence` is a core trade-off between **precision** (fewer false positives) and **recall** (catching more real issues). It is a frequently tested concept on the AIF-C01 exam.

---

## 📁 Files

```
02-rekognition/
├── rekognition_content_moderator.ipynb
├── screenshots/
│   └── photo_content_moderator.png
└── README.md
```

---

## 💡 What I Learned

The biggest insight from this project was understanding the difference between **managed AI services** and **custom ML models**. With Rekognition, I called a single API and got production-quality computer vision results in under 2 seconds — no training data, no model deployment, no cleanup. It is the core trade-off the AIF-C01 exam tests: use a pre-built service like Rekognition when your use case fits, use SageMaker when you need a custom model for a unique problem.

---

