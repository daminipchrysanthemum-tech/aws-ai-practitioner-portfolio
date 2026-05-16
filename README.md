## 🏠 House Price Predictor — Amazon SageMaker

> Predicts California median house prices using XGBoost trained and deployed on Amazon SageMaker. Built as part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshots

### Model Predictions & RMSE
<img width="1902" height="845" alt="House Price Predictor 01" src="https://github.com/user-attachments/assets/8687ac4d-95e0-4578-bc0d-1431812b13e9" />


### Feature Engineering — 4 New Features Verified
<img width="1897" height="887" alt="House Price Predictor - 02" src="https://github.com/user-attachments/assets/99426783-ca5a-4cb7-b02c-8e22955863cc" />

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
House Price Predictor 
├── california_housing.ipynb
├── screenshots/
│   ├── house_price_predictor-01.png
│   └── house price_predictor_02.png
└── results/
    └── predictions_v1.csv
```

---

## 💡 What I Learned

Working through this project taught me how AWS separates the training and deployment steps — you pay for compute only while it's running. The most surprising thing was how much feature engineering matters: adding just 4 derived columns could cut RMSE by 15–25% without changing the algorithm at all. It maps directly to what the AIF-C01 exam tests in the ML fundamentals domain.

---

## 🔍 Photo Content Moderator — Amazon Rekognition

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

All 4 images were correctly identified as safe content with `MinConfidence=10`, demonstrating that the moderator works end-to-end and correctly returns Clean for non-flagged images.

---

## 🧠 What This Project Does

Uses **Amazon Rekognition's `detect_moderation_labels` API** to scan images stored in S3 and classify them as safe or flagged. For each image, Rekognition returns a list of moderation labels (e.g., Explicit Nudity, Violence, Drugs, Alcohol, Gambling) with confidence scores from 0–100. The notebook loops through a batch of images and produces a final summary report as a pandas DataFrame.

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
If labels are empty → "Clean"
If labels found → "Flagged: <category>"
        ↓
Results compiled into a pandas DataFrame report
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

## 🧠 Product Review Analyzer — Amazon Comprehend

> Analyzes product reviews using Amazon Comprehend's pre-trained NLP models to detect sentiment, extract key phrases, and identify named entities (brands, locations, dates). Built as part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshot

### Final Review Report & Sentiment Breakdown
<img width="1917" height="891" alt="Product Review Analyzer" src="https://github.com/user-attachments/assets/1c541085-50a1-479c-869e-c90c7751f953" />


---

## 📊 Results

| Review | Product | Sentiment | Confidence |
|--------|---------|-----------|------------|
| 1 | Wireless Headphones | Positive | 100.0% |
| 2 | Wireless Headphones | Negative | 100.0% |
| 3 | Wireless Headphones | Mixed | 98.6% |
| 4 | Coffee Maker | Positive | 100.0% |
| 5 | Coffee Maker | Negative | 100.0% |
| 6 | Running Shoes | Positive | 100.0% |
| 7 | Running Shoes | Mixed | 99.2% |
| 8 | Laptop Stand | Positive | 100.0% |
| 9 | Laptop Stand | Mixed | 99.8% |
| 10 | Bluetooth Speaker | Positive | 99.5% |

### Sentiment Breakdown
```
POSITIVE   █████  5 reviews (50%)
MIXED      ███    3 reviews (30%)
NEGATIVE   ██     2 reviews (20%)
```

---

## 🧠 What This Project Does

Uses **Amazon Comprehend's NLP APIs** to analyze 10 realistic product reviews across 4 product categories. For each review, the notebook detects overall sentiment (Positive/Negative/Mixed/Neutral) with confidence scores, extracts the top key phrases, and identifies named entities such as brand names, locations, and dates. Results are compiled into a pandas DataFrame report with a summary of the sentiment breakdown.

---

## 🔑 How Comprehend Compares to Other Services

| | SageMaker | Rekognition | Comprehend |
|--|-----------|-------------|------------|
| Input | Tabular data | Images in S3 | Plain text |
| Setup | Training + endpoint | S3 upload | Nothing — text direct |
| Cleanup | Delete endpoint | Nothing | Nothing |
| Pricing | Per instance-hour | Per image | Per 100 characters |
| Use case | Custom ML models | Computer vision | Text understanding |

---

## ⚙️ How It Works

```
Plain text review passed directly to API
              ↓
┌─────────────────────────────────┐
│   detect_sentiment()            │
│   → POSITIVE / NEGATIVE /       │
│     MIXED / NEUTRAL + scores    │
├─────────────────────────────────┤
│   detect_key_phrases()          │
│   → Top topics mentioned        │
│     with confidence scores      │
├─────────────────────────────────┤
│   detect_entities()             │
│   → ORGANIZATION, LOCATION,     │
│     PERSON, DATE, QUANTITY      │
└─────────────────────────────────┘
              ↓
   Results compiled into
   pandas DataFrame report
```

---

## 🔍 Entity Detection Highlights

Comprehend correctly identified real-world entities from the review text:

| Review | Entity Detected | Type |
|--------|----------------|------|
| 7 — Running Shoes | Nike | ORGANIZATION |
| 7 — Running Shoes | New York | LOCATION |
| 10 — Bluetooth Speaker | Miami | LOCATION |

---

## 💻 Core Code

```python
import boto3
import pandas as pd

comprehend = boto3.client("comprehend", region_name="us-east-1")

def analyze_review(review):
    text = review["text"]

    sentiment_response = comprehend.detect_sentiment(
        Text=text, LanguageCode="en"
    )
    keyphrases_response = comprehend.detect_key_phrases(
        Text=text, LanguageCode="en"
    )

    sentiment  = sentiment_response["Sentiment"]
    scores     = sentiment_response["SentimentScore"]
    top_score  = round(scores[sentiment.title()] * 100, 1)
    top_phrases = [
        p["Text"] for p in
        sorted(keyphrases_response["KeyPhrases"],
               key=lambda x: x["Score"], reverse=True)[:3]
    ]

    return {
        "product":    review["product"],
        "sentiment":  sentiment,
        "confidence": f"{top_score}%",
        "key_phrases": " | ".join(top_phrases)
    }

results = [analyze_review(r) for r in reviews]
df = pd.DataFrame(results)
display(df)
```

---

## 🧪 AWS Concepts Practiced

- ✅ `detect_sentiment` — Positive / Negative / Mixed / Neutral
- ✅ `detect_key_phrases` — most important topics in text
- ✅ `detect_entities` — brands, locations, dates, people
- ✅ Confidence scores and how to interpret them
- ✅ Serverless / pay-per-call NLP (no infrastructure)
- ✅ IAM permission fix — `ComprehendFullAccess` policy
- ✅ Principle of least privilege (IAM concept)
- ✅ pandas DataFrame for results reporting
- ✅ Batch processing multiple text inputs

---

## 📝 AIF-C01 Exam Topics It Covers

| Topic | Domain |
|-------|--------|
| NLP use cases — sentiment, entities, key phrases | Domain 1: Fundamentals of AI/ML |
| Pre-built AI services vs custom models | Domain 1: Fundamentals of AI/ML |
| Confidence scores and thresholds | Domain 1: Fundamentals of AI/ML |
| Serverless AI pricing (pay-per-call) | Domain 2: Generative AI |
| IAM policies and least privilege | Domain 5: Security & Governance |
| When to use Comprehend vs SageMaker | Domain 1: Fundamentals of AI/ML |
| Responsible AI — content understanding | Domain 4: Responsible AI |

---

## 💡 Key Concept: Sentiment Confidence Scores

Comprehend returns 4 scores per review, always summing to 1.0:

```python
{
  "Positive": 0.986,
  "Negative": 0.002,
  "Mixed":    0.011,
  "Neutral":  0.001
}
# → Sentiment: POSITIVE (98.6% confident)
```

A score above 0.95 is high confidence. Scores between 0.50 and 0.75 indicate
weaker or more ambiguous sentiment — worth flagging for human review.
This threshold concept is frequently tested on the AIF-C01 exam.

---

## 🚀 How to Run

### Prerequisites
- AWS account with SageMaker Studio access
- IAM role with `ComprehendFullAccess` policy attached

### Steps
1. Open `comprehend_review_analyzer.ipynb` in SageMaker Studio
2. Run all cells top to bottom
3. View the Final Review Report table and Sentiment Breakdown

### No cleanup needed
Comprehend is fully serverless. No endpoints, no instances,
nothing to shut down. Cost: ~$0.0001 per 100 characters of text.

---

## 📁 Files

```
03-comprehend/
├── comprehend_review_analyzer.ipynb
├── review_analysis_results.csv
├── screenshots/
│   └── comprehend_review_analyzer.png
└── README.md
```

---

## 💡 What I Learned

The biggest insight was understanding mixed sentiment — reviews that are genuinely both positive and negative at the same time (good sound quality but uncomfortable fit). Comprehend handles this with a dedicated MIXED label rather than forcing a binary choice. It matters in real-world applications like product feedback dashboards, where nuanced sentiment is more useful than a simple thumbs up/down. I also learned that IAM permissions must be explicitly granted per service — my SageMaker role needed ComprehendFullAccess added before any Comprehend API calls would work, which is a direct example of the Principle of least privilege tested on AIF-C01.

---

## 🤖 Document Q&A Bot — Amazon Bedrock

> Builds a multi-turn conversational Q&A bot that answers questions from any document using Amazon Bedrock's Converse API and Claude 4.5 Haiku. Built as part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshots

### Notebook Code — Document & Questions Setup
<img width="1902" height="573" alt="Document Q A Bot - 01" src="https://github.com/user-attachments/assets/99bf6833-fe95-49c6-8b02-8127106e51d3" />


### Bot Output — 5 Questions Answered with Multi-Turn Memory
<img width="1901" height="683" alt="Document Q A Bot - 02" src="https://github.com/user-attachments/assets/17b724ad-16de-46b7-8ab6-69755f14e635" />

---

## 📊 Results

| Question | Correct Answer Given |
|----------|---------------------|
| Which service analyzes customer review sentiment? | ✅ Comprehend |
| Difference between SageMaker and Bedrock? | ✅ Full comparison with bullet points |
| Which services are fully serverless? | ✅ Comprehend + Bedrock |
| How does pricing differ across services? | ✅ Per instance-hour vs per API call breakdown |
| Which service detects faces in photos? | ✅ Rekognition |

Total exchanges: 5 — Session complete ✅

---

## 🧠 What This Project Does

Uses **Amazon Bedrock's Converse API** with **Claude 4.5 Haiku** to build a document-aware Q&A bot. A study document is injected into the system prompt as context, and the bot answers questions about it using multi-turn conversation memory. Each exchange is appended to `conversation_history` so the model builds on previous answers throughout the session — demonstrating the RAG pattern and multi-turn conversation management.

---

## 🆕 2025 Updates — What Changed

| Old behavior  | New behavior  (2025) |
|--------------|---------------------|
| Manual "Enable model" per model | All models auto-enabled by default |
| Model Access page in console | Retired Sep 29 2025 — use Model Catalog instead |
| `InvokeModel` API per-model format | Unified `Converse` API works across all models |
| Separate request format per provider | Single messages array format for all models |

---

## ⚙️ How It Works

```
Study document injected as system prompt context
                    ↓
User question added to conversation_history
                    ↓
bedrock.converse(modelId, system, messages)
                    ↓
Claude 4.5 Haiku answers from document context only
                    ↓
Answer appended to conversation_history
                    ↓
Next question builds on previous answers (multi-turn)
```

---

## 💻 Core Code

```python
import boto3

bedrock   = boto3.client("bedrock-runtime", region_name="us-west-2")
MODEL_ID  = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

def chat_with_document(document, questions):
    conversation_history = []
    system_prompt = f"""You are a helpful study assistant.
Answer questions based on this document only:

{document}

If the answer is not in the document, say so clearly."""

    for question in questions:
        conversation_history.append({
            "role": "user",
            "content": [{"text": question}]
        })

        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": system_prompt}],
            messages=conversation_history,
            inferenceConfig={"maxTokens": 512, "temperature": 0.3}
        )

        answer = response["output"]["message"]["content"][0]["text"]
        conversation_history.append({
            "role": "assistant",
            "content": [{"text": answer}]
        })

        print(f"Q: {question}")
        print(f"A: {answer}\n")
```

---

## 🔑 Key Concepts Demonstrated

### Converse API message format
```python
messages = [
    {"role": "user",      "content": [{"text": "Your question"}]},
    {"role": "assistant", "content": [{"text": "Previous answer"}]},
    {"role": "user",      "content": [{"text": "Follow-up question"}]}
]
```

### Inference parameters explained
```python
inferenceConfig={
    "maxTokens":  512,  # max output length
    "temperature": 0.3, # lower = more focused, less random
    "topP":        0.9  # nucleus sampling threshold
}
```

---

## 🧪 AWS Concepts Practiced

- ✅ Amazon Bedrock `converse()` API (2025 Converse API)
- ✅ Claude 4.5 Haiku foundation model via Bedrock
- ✅ System prompts for controlling model behavior
- ✅ Multi-turn conversation history management
- ✅ RAG pattern — document injected as context
- ✅ Inference parameters: temperature, maxTokens, topP
- ✅ Pay-per-token serverless pricing model
- ✅ IAM permissions — `AmazonBedrockFullAccess` policy
- ✅ Anthropic use case form — one-time account setup
- ✅ Model catalog — new 2025 console experience
- ✅ Amazon Nova Lite as an alternative AWS-native model

---

## 📝 AIF-C01 Exam Topics It Covers

| Topic | Domain |
|-------|--------|
| Foundation models and how to access them | Domain 2: Generative AI |
| Prompt engineering and system prompts | Domain 2: Generative AI |
| RAG — retrieval augmented generation | Domain 2: Generative AI |
| Temperature, tokens, topP parameters | Domain 2: Generative AI |
| Multi-turn conversation and context window | Domain 2: Generative AI |
| Pre-built vs custom models trade-offs | Domain 1: Fundamentals of AI/ML |
| Serverless AI pricing (pay-per-token) | Domain 2: Generative AI |
| IAM permissions for AI services | Domain 5: Security & Governance |
| Responsible AI — system prompt guardrails | Domain 4: Responsible AI |

---

## 🚀 How to Run

### Prerequisites
- AWS account with SageMaker Studio access
- IAM role with `AmazonBedrockFullAccess` policy attached
- One-time Anthropic use case form submitted via Bedrock console

### Enable Anthropic Claude (one-time)
1. Go to AWS Console → Amazon Bedrock → Model catalog
2. Search for Claude 4.5 Haiku → Open in playground
3. Submit the Anthropic use case form when prompted
4. Access granted immediately — never needed again

### Steps
1. Open `bedrock_document_qa_bot.ipynb` in SageMaker Studio
2. Run all cells top to bottom
3. Replace `my_own_document` with any text you want to interrogate
4. Update `my_questions` with your own questions

### No cleanup needed
Bedrock is fully serverless. No endpoints, no instances, nothing
to shut down. Cost for this entire project: under $0.05.

---

## 📁 Files

```
04-bedrock/
├── bedrock_document_qa_bot.ipynb
├── screenshots/
│   ├── document_qa_bot_01.png
│   └── document_qa_bot_02.png
└── README.md
```

---

## 💡 What I Learned

The most important insight from this project was understanding the **RAG pattern** — injecting a document as context so the model answers from YOUR content rather than its general training data. It's how real enterprise AI applications work: a knowledge base of company documents is retrieved and injected into the prompt at query time, keeping answers accurate and up to date. I also learned the difference between `temperature` (how creative vs focused the model is) and `maxTokens` (how long the response can be) — both are core AIF-C01 exam concepts. The 2025 Converse API made this significantly simpler than older tutorials: one unified code pattern works across Claude, Nova, Llama, and Mistral without changing anything except the `modelId`.

---

## 🤖 Resume Q&A Chatbot — Amazon Q Business

> A live AI-powered chatbot that answers recruiter questions about my AWS
> projects, technical skills, and experience. Built on Amazon Q Business
> with anonymous access — no login required to interact with it.
> Part of my AWS Certified AI Practitioner (AIF-C01) hands-on portfolio.

---

## 📸 Screenshots

### Chatbot welcome screen — anonymous Guest access
<img width="1079" height="658" alt="Amazon Q Business - Resume Q A chatbot" src="https://github.com/user-attachments/assets/2b3d7e86-c3c4-4338-9b6a-9c9b92dadb4e" />

### All 7 recruiter questions answered correctly
<img width="833" height="803" alt="Amazon Q Business - Resume Q A chatbot (Questions)" src="https://github.com/user-attachments/assets/3129c2af-91c5-4b5f-90b1-4365048481a1" />

---

## 💬 Live demo results

Every recruiter question answered correctly from indexed documents:

| Question asked | Answer pulled from docs |
|----------------|------------------------|
| What AWS AI services have you worked with? | SageMaker, Rekognition, Comprehend, Bedrock, Amazon Q Business |
| Describe the SageMaker project and results | XGBoost on California Housing, 4 derived features, RMSE $47,521 |
| What is your RMSE score? | $47,521 — exact figure cited with source |
| What programming languages and tools? | Python, boto3, pandas, scikit-learn |
| Studying for any AWS certifications? | AIF-C01 is currently in progress |
| What is RAG, and have you implemented it? | Implemented in Project 4 via Bedrock Converse API with Claude 4.5 Haiku |
| Which project are you most proud of? | Bedrock Document Q&A Bot — demonstrates full RAG architecture |

---

## 🧠 What this project does

Deploys an **Amazon Q Business application** with anonymous user access
that indexes three portfolio documents stored in Amazon S3. When a user
asks a question, Q Business searches its native index using semantic search,
retrieves the most relevant document chunks and generates a grounded answer
with source citations. No authentication is required — anyone with the link
can interact instantly.

---

## 🏗️ Architecture

```
3 portfolio documents (.txt files)
           ↓
   Amazon S3 bucket (us-east-1)
           ↓
   Amazon Q Business
   ├── S3 data source connector (syncs documents)
   ├── Native index (vector store — auto-provisioned)
   ├── Native retriever (semantic search — auto-created)
   └── Anonymous web experience (public chatbot UI)
           ↓
   Recruiter opens link → types question
           ↓
   Retriever finds relevant chunks from the index
           ↓
   The Foundation model generates a grounded answer
   with source citations [1] [2] [3]
```

---

## 📄 Documents indexed

| File | Contents |
|------|----------|
| `resume.txt` | Full resume — experience, education, skills |
| `projects.txt` | All 5 portfolio projects with results and technical details |
| `skills.txt` | AWS services, tools, certifications in progress |

---

## ⚙️ AWS services and concepts used

- **Amazon Q Business** — application, native index, retriever, web experience
- **Amazon S3** — document storage and data source connector
- **IAM** — service roles for Q Business and S3 read access
- **Anonymous access** — public chatbot with no IAM Identity Center required
- **Native index** — managed vector database for document embeddings
- **Sync jobs** — automated document crawling and re-indexing

---

## 🧪 AWS concepts practiced

- ✅ Amazon Q Business anonymous application setup
- ✅ Native index creation (Starter tier, 1 unit)
- ✅ Retriever auto-creation alongside index
- ✅ S3 data source connector configuration
- ✅ Document sync jobs and troubleshooting
- ✅ Full RAG pipeline — document → embed → index → retrieve → generate
- ✅ Source citations in grounded responses
- ✅ Web experience customization (title, subtitle, welcome message)
- ✅ Consumption-based billing vs provisioned pricing
- ✅ Anonymous vs authenticated (IAM Identity Center) access patterns
- ✅ IAM role permissions for cross-service access (Q Business → S3)

---

## 📝 AIF-C01 Exam Topics It Covers

| Topic | Domain |
|-------|--------|
| RAG architecture — retrieve, augment, generate | Domain 3: Applications of Foundation Models |
| Foundation models in fully managed services | Domain 2: Fundamentals of Generative AI |
| Amazon Q Business use cases and architecture | Domain 3: Applications of Foundation Models |
| Vector databases and semantic search | Domain 3: Applications of Foundation Models |
| Grounded responses and source citations | Domain 4: Responsible AI |
| Hallucination reduction via RAG | Domain 4: Responsible AI |
| Anonymous vs authenticated access | Domain 5: Security & Governance |
| Consumption-based AI pricing model | Domain 2: Fundamentals of Generative AI |

---

## 🔑 Key concept: RAG vs direct prompting

This project shows the difference between two approaches:

**Project 4 (Bedrock) — manual RAG:**
```python
# You manually inject context into every prompt
response = bedrock.converse(
    messages=[{"role": "user", "content": [{"text":
        f"Context: {document}\n\nQuestion: {question}"
    }]}]
)
```

**Project 5 (Amazon Q) — managed RAG:**
```
Upload docs to S3 → Q Business handles everything automatically:
chunking → embedding → indexing → retrieval → generation → citations
```

Same RAG architecture. Amazon Q is the fully managed, production-scale
version. This distinction is directly tested on the AIF-C01 exam.

---

## 🚀 How to recreate this project

### Prerequisites
- AWS account in us-east-1
- 3 portfolio documents saved as .txt files
- S3 bucket in us-east-1

### Steps
1. Upload .txt files to the S3 bucket
2. Open Amazon Q Business → Create application
3. User access: **Anonymous** + enable **Web experience**
4. Data sources → **Add index** → Starter, 1 unit (retriever auto-created)
5. **Add data source** → Amazon S3 → Browse to bucket → Sync now
6. Web experience settings → edit title, subtitle, welcome message
7. Share → Create URL (60 min session) → copy for portfolio

### Regenerating the demo link
The anonymous URL expires per session. Regenerate anytime:
Console → Application → Web experience settings → Share → Create URL

---

## 📁 Files

```
05-amazon-q/
├── README.md
├── sample-docs/
│   ├── projects_template.txt    ← template for your own projects
│   └── skills_template.txt      ← template for skills section
└── screenshots/
    ├── amazon_q_business_resume_q&a_chatbot.png
    └── amazon_q_business_resume_q&a_chatbot_questions.png
```

---

## 💡 What I learned

The most valuable insight from this project was seeing managed RAG in action compared to the manual RAG I built in Project 4. Amazon Q Business handles
document chunking, embedding generation, vector indexing, semantic retrieval, and grounded response generation automatically — the same pipeline that would
take hundreds of lines of code to build manually with LangChain or LlamaIndex.

The anonymous access feature (launched April 2025) transforms this from a technical exercise into a real recruiter-facing tool. The anonymous access feature (launched April 2025) transforms this from a technical exercise into a real recruiter-facing tool. Anyone can open the link and ask the bot questions about my experience without an AWS account, making it the most portfolio-impactful project of the five.

---

