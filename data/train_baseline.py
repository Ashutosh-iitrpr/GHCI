import pandas as pd #type: ignore
import numpy as np #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, f1_score #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import joblib #type: ignore
import os

# -------------------------------
# 1. Load and clean data
# -------------------------------
df = pd.read_csv("data/synthetic_transactions.csv")

print("✅ Dataset loaded.")
print(df.head(), "\n")

# Basic cleaning
df['transaction_text'] = df['transaction_text'].astype(str).str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['transaction_text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

# -------------------------------
# 2. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 3. Train classifier
# -------------------------------
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_vec, y_train)

# -------------------------------
# 4. Evaluate
# -------------------------------
y_pred = model.predict(X_test_vec)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n🏁 Macro F1 Score: {macro_f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Save model & vectorizer
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to /models/")
