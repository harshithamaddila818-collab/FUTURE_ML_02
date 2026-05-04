import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# =========================
# DOWNLOAD STOPWORDS
# =========================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# RULE-BASED PRIORITY BOOST
# =========================
def smart_priority(text, model_pred):
    text = text.lower()

    if "refund" in text or "money deducted" in text:
        return "Critical"
    if "failed" in text or "error" in text or "not working" in text:
        return "High"

    return model_pred

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("tickets.csv")
print("✅ Dataset Loaded")

# =========================
# PREPARE DATA
# =========================
df['text'] = df['Ticket Subject'] + " " + df['Ticket Description']

df = df.rename(columns={
    'Ticket Type': 'category',
    'Ticket Priority': 'priority'
})

df = df[['text', 'category', 'priority']].dropna()

print("\n✅ Cleaned Data:")
print(df.head())

# =========================
# APPLY CLEANING
# =========================
df['clean_text'] = df['text'].apply(clean_text)
print("\n✅ Text Cleaned")

# =========================
# CATEGORY MODEL
# =========================
X = df['clean_text']
y_cat = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
)

pipe_cat = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LinearSVC())
])

pipe_cat.fit(X_train, y_train)
pred_cat = pipe_cat.predict(X_test)

print("\n===== CATEGORY MODEL (SVM) =====")
print(classification_report(y_test, pred_cat))

# =========================
# PRIORITY MODEL
# =========================
y_pri = df['priority']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pri, test_size=0.2, stratify=y_pri, random_state=42
)

pipe_pri = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LinearSVC())
])

pipe_pri.fit(X_train, y_train)
pred_pri = pipe_pri.predict(X_test)

print("\n===== PRIORITY MODEL (SVM) =====")
print(classification_report(y_test, pred_pri))

# =========================
# PREDICTION FUNCTION
# =========================
def predict_ticket(text):
    cleaned = clean_text(text)

    category = pipe_cat.predict([cleaned])[0]
    priority = pipe_pri.predict([cleaned])[0]

    # apply business logic
    priority = smart_priority(text, priority)

    return category, priority

# =========================
# TEST
# =========================
test_ticket = "Payment failed and money deducted but order not confirmed"

cat, pri = predict_ticket(test_ticket)

print("\n🔮 Prediction:")
print("Ticket:", test_ticket)
print("Category:", cat)
print("Priority:", pri)