# predict_transaction.py
import os
import json
import joblib #type: ignore
import pandas as pd #type: ignore
import numpy as np #type: ignore

MODEL_PATH = "models/baseline_model.pkl"
VECT_PATH = "models/tfidf_vectorizer.pkl"
FEEDBACK_PATH = "data/user_feedback.csv"

CONFIDENCE_THRESHOLD = 0.6
TOP_K = 3

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError(f"Model/vectorizer missing. Train with retrain_with_feedback.py first.\nExpected: {MODEL_PATH}, {VECT_PATH}")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
print("✅ Model and vectorizer loaded successfully!")

def predict_transaction(texts, threshold=CONFIDENCE_THRESHOLD, top_k=TOP_K):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [t.strip() for t in texts]
    X_vec = vectorizer.transform(cleaned)
    probs = model.predict_proba(X_vec)
    classes = list(model.classes_)

    rows = []
    for i, txt in enumerate(texts):
        prob_arr = probs[i]
        best_idx = int(np.argmax(prob_arr))
        pred_cat = classes[best_idx]
        conf = float(prob_arr[best_idx])
        review_flag = "✅" if conf >= threshold else "⚠️ Low confidence"
        topk_idx = list(np.argsort(prob_arr)[::-1][:top_k])
        top_probs = { classes[j]: float(round(prob_arr[j], 6)) for j in topk_idx }

        rows.append({
            "transaction_text": txt,
            "predicted_category": pred_cat,
            "confidence": conf,
            "review_required": review_flag,
            "true_category": "",      # user will edit this field
            "source": "user",         # mark these rows as user-sourced
            "top_probs_json": json.dumps(top_probs),
            "timestamp": pd.Timestamp.now().isoformat()
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    df.to_csv(FEEDBACK_PATH, index=False)
    print(f"📝 Predictions saved to {FEEDBACK_PATH} (overwrite).")
    return df

if __name__ == "__main__":
    sample_inputs = [
        "Payment bill upi to ashking",
        "UPI payment to nykaa",
        "Electricity Bill Payment BESCOM",
        "Booking IRCTC Train Ticket",
        "Online purchase Amazon order #445"
    ]

    df = predict_transaction(sample_inputs)
    print("\n🔍 Predictions:")
    print(df[["transaction_text", "predicted_category", "confidence", "review_required"]].to_string(index=False))
