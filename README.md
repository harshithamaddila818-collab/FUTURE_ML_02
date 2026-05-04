# 🎫 Support Ticket Classification & Prioritization

## 📌 Overview
This project builds an intelligent system that automatically analyzes customer support tickets and performs two key tasks:
1. Classifies tickets into categories (e.g., Technical Issue, Billing Inquiry, Refund Request)
2. Assigns a priority level (Critical, High, Medium, Low)

The goal is to help businesses reduce manual effort, speed up response time, and improve customer satisfaction.

---

## 🚀 Features
- Text preprocessing (lowercasing, stopword removal, noise cleaning)
- Feature extraction using TF-IDF (including n-grams)
- Machine learning models for:
  - Ticket category classification
  - Priority prediction
- Rule-based enhancement for urgent tickets (e.g., payment failures, refunds)
- Interactive web app using Streamlit

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- NLTK
- Streamlit

---

## 📊 Model Performance
- Category Classification Accuracy: ~20–25%
- Priority Prediction Accuracy: ~24–27%

> Note: Accuracy is limited due to the dataset containing repetitive and templated ticket text.

---

## 💡 Business Impact
- Automates ticket categorization
- Helps prioritize urgent issues faster
- Reduces workload on support teams
- Improves response efficiency

---
## How to run
```bash
pip install -r requirements.txt
python python.py
streamlit run app.py
