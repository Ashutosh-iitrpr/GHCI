import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 1. Load Model and Vectorizer
# -----------------------------
model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

pipeline = make_pipeline(vectorizer, model)
class_names = model.classes_

print("✅ Model and vectorizer loaded successfully!\n")

# -----------------------------
# 2. LIME Explanation
# -----------------------------
def explain_with_lime(text, num_features=8):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=num_features)
    print(f"\n🔍 Transaction: {text}")
    print(f"Predicted Category: {pipeline.predict([text])[0]}")
    html_path = "lime_explanation.html"
    exp.save_to_file(html_path)
    print(f"✅ LIME explanation saved to {html_path}")
    return exp

# -----------------------------
# 3. SHAP Explanation
# -----------------------------
def explain_with_shap(sample_texts, max_display=10):
    print("\n⚙️ Computing SHAP values (this may take a few seconds)...")

    # 1. Define a callable model function
    f = lambda x: pipeline.predict_proba(x)

    # 2. Use a simple text masker (no tokenizer dict)
    masker = shap.maskers.Text()

    # 3. Initialize SHAP explainer (model-agnostic KernelExplainer)
    explainer = shap.Explainer(f, masker)

    # 4. Compute SHAP values
    shap_values = explainer(sample_texts)

    # 5. Display local explanations for each sample
    for i, text in enumerate(sample_texts):
        print(f"\n🧾 Transaction {i+1}: {text}")
        shap.plots.text(shap_values[i])

# -----------------------------
# 4. Example usage
# -----------------------------
if __name__ == "__main__":
    sample_texts = [
        "UPI payment to zomato@okhdfcbank",
        "Electricity Bill Payment BESCOM 1200 Rs",
        "Online purchase Amazon order #394829"
    ]
    
    # LIME example (interactive HTML in notebook)
    explain_with_lime(sample_texts[1])
    
    # SHAP example (local & global explanations)
    explain_with_shap(sample_texts)
