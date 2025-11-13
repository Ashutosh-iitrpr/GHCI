# retrain_with_feedback.py
"""
Retrain pipeline that:
 - reads config/taxonomy.yaml
 - accepts feedback rows whose true_category exists in taxonomy
 - pending rows (unknown categories) are appended to data/pending_new_categories.csv
 - generates synthetic examples:
     * admin-sourced feedback -> more variants (SYNTH_ADMIN) & higher weight
     * user-sourced feedback  -> fewer variants (SYNTH_USER) & lower weight
 - trains TF-IDF + LogisticRegression using sample weights
"""

import os
import re
import random
import pandas as pd #type: ignore
import numpy as np #type: ignore
import joblib #type: ignore
from yaml import safe_load #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import f1_score, classification_report #type: ignore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# CONFIG
MAIN_DATA = os.path.join(ROOT, "data", "synthetic_transactions.csv")
FEEDBACK  = os.path.join(ROOT, "data", "user_feedback.csv")
PENDING   = os.path.join(ROOT, "data", "pending_new_categories.csv")
TAX_PATH  = os.path.join(ROOT, "config", "taxonomy.yaml")
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
VECT_PATH  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SEEDS_PATH = os.path.join(ROOT, "data", "taxonomy_synonym_seeds.csv")
SUGGEST_PATH = os.path.join(ROOT, "data", "merchant_suggestions.csv")

# Augmentation / weight knobs
SYNTH_ADMIN = 20
SYNTH_ADMIN_SYNONYM = 15
SYNTH_USER = 6
WEIGHT_MAIN = 1.0
WEIGHT_ADMIN = 1.5
WEIGHT_USER = 0.3

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def read_taxonomy(path=TAX_PATH):
    if not os.path.exists(path):
        return {"categories": {}}
    with open(path, "r", encoding="utf-8") as f:
        return safe_load(f) or {"categories": {}}

def extract_merchant_token(text):
    text = str(text).lower()
    m = re.search(r"(?:to|at)\s+([a-z0-9@._-]+)", text)
    if m:
        token = m.group(1)
    else:
        toks = re.findall(r"[a-z0-9@._-]+", text)
        toks = [t for t in toks if len(re.sub(r'[^a-z]', '', t)) >= 3]
        token = toks[0] if toks else None
    if token:
        token = re.sub(r'@.*$', '', token)
        token = re.sub(r'\..*$', '', token)
        token = re.sub(r'[^a-z0-9]', '', token)
        return token or None
    return None

def generate_variants(merchant, amount=None, n=8):
    templates = [
        f"POS {merchant} {int(amount) if amount else random.randint(50,1500)} TXN",
        f"UPI payment to {merchant}@okhdfcbank",
        f"PURCHASE {merchant} Ref#{random.randint(1000,9999)}",
        f"Online payment to {merchant}",
        f"DEBIT CARD {merchant.upper()} {int(amount) if amount else random.randint(50,1500)}",
        f"PAYMENT TO {merchant} VIA NETBANKING",
        f"Txn {random.randint(10000,99999)}: {merchant} {random.randint(50,1500)}",
        f"IMPS transfer to {merchant}",
        f"{merchant} *{random.randint(1000,9999)} POS Txn",
        f"{merchant} order #{random.randint(100,9999)}"
    ]
    variants = []
    for i in range(n):
        base = random.choice(templates)
        if random.random() < 0.2:
            base = base.replace(merchant, merchant.upper())
        if random.random() < 0.15:
            pos = random.randint(0, max(0, len(base)-1))
            base = base[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + base[pos+1:]
        if random.random() < 0.12:
            base = base + f" REF{random.randint(10,9999)}"
        variants.append(base)
    return variants

def main():
    print("📥 Loading main dataset...")
    main_df = pd.read_csv(MAIN_DATA)
    taxonomy = read_taxonomy()
    allowed_categories = set([c.strip().lower() for c in taxonomy.get('categories', {}).keys()])

    if not os.path.exists(FEEDBACK):
        print("⚠️ No feedback file found. Exiting.")
        return

    fb = pd.read_csv(FEEDBACK).dropna(how='all', axis=1)
    for c in ['transaction_text', 'true_category']:
        if c not in fb.columns:
            fb[c] = ""
    # default source = user if not present
    if "source" not in fb.columns:
        fb["source"] = "user"

    fb['transaction_text'] = fb['transaction_text'].astype(str).str.strip()
    fb['true_category'] = fb['true_category'].astype(str).str.strip().str.lower()
    fb = fb[fb['true_category'].notna() & (fb['true_category'] != "") & (fb['true_category'] != "nan")]

    # ---------- build accepted / pending from user feedback ----------
    accepted = fb[fb['true_category'].isin(allowed_categories)].copy()
    pending  = fb[~fb['true_category'].isin(allowed_categories)].copy()

    # append pending to PENDING file (dedup)
    if not pending.empty:
        if os.path.exists(PENDING):
            prev = pd.read_csv(PENDING).dropna(how='all', axis=1)
            combined_pending = pd.concat([prev, pending], ignore_index=True).drop_duplicates(subset=['transaction_text','true_category'])
        else:
            combined_pending = pending.copy()
        os.makedirs(os.path.dirname(PENDING) or ".", exist_ok=True)
        combined_pending.to_csv(PENDING, index=False)
        print(f"🕒 {len(pending)} feedback rows held in pending_new_categories.csv (total pending: {len(combined_pending)})")

    # -------------------------
    # Admin synonym seeds (trusted admin-added merchants)
    # -------------------------
    # helper: read unused seeds (merchant, category, used)
    def read_unused_synonym_seeds(path=SEEDS_PATH):
        if not os.path.exists(path):
            return pd.DataFrame(columns=["merchant","category","used"])
        df = pd.read_csv(path).dropna(how='all', axis=1)
        for c in ["merchant","category","used"]:
            if c not in df.columns:
                df[c] = "" if c != "used" else 0
        df["merchant"]  = df["merchant"].astype(str).str.strip().str.lower()
        df["category"]  = df["category"].astype(str).str.strip().str.lower()
        df["used"]      = df["used"].fillna(0).astype(int)
        # keep only unused
        return df[(df["merchant"]!="") & (df["category"]!="") & (df["used"]==0)].reset_index(drop=True)

    def mark_seeds_used(merchants, categories, path=SEEDS_PATH):
        if not os.path.exists(path): return
        df = pd.read_csv(path).dropna(how='all', axis=1)
        mask = df["merchant"].astype(str).str.lower().isin(merchants) & df["category"].astype(str).str.lower().isin(categories)
        df.loc[mask, "used"] = 1
        df.to_csv(path, index=False)

    # load unused seeds and keep only those with a valid category in current taxonomy
    seed_df = read_unused_synonym_seeds()
    seed_df = seed_df[ seed_df["category"].isin(allowed_categories) ].reset_index(drop=True)

    # If there's no accepted user rows AND no admin seeds, nothing to incorporate -> exit
    if accepted.empty and seed_df.empty:
        print("ℹ️ No accepted feedback rows and no admin seeds to incorporate. Exiting.")
        return

    # otherwise we proceed: accepted may be empty but seed_df can provide admin synthetic rows
    print(f"📥 Accepted feedback rows: {len(accepted)}, Admin seeds to consume: {len(seed_df)}")

    
    def load_known_synonyms(tax):
        syn = set()
        for cat, data in tax.get("categories", {}).items():
            for m in data.get("synonyms", []) or []:
                syn.add(str(m).strip().lower())
        return syn
    
    print("RETRAIN_START: loading training pipeline")

    known_synonyms = load_known_synonyms(taxonomy)
    suggest_rows = []

    for _, row in accepted.iterrows():
        m = extract_merchant_token(row['transaction_text'])
        if m and m not in known_synonyms:
            suggest_rows.append({"merchant": m, "suggested_category": row['true_category']})

    if suggest_rows:
        new_suggest = pd.DataFrame(suggest_rows).drop_duplicates()
        if os.path.exists(SUGGEST_PATH):
            prev = pd.read_csv(SUGGEST_PATH).dropna(how="all", axis=1)
            merged = pd.concat([prev, new_suggest], ignore_index=True).drop_duplicates()
        else:
            merged = new_suggest
        merged.to_csv(SUGGEST_PATH, index=False)
        print(f"📌 Logged {len(new_suggest)} merchant suggestions to {SUGGEST_PATH}")

    # load seeds
    seed_df = read_unused_synonym_seeds()
    # keep only seeds whose category exists in taxonomy
    seed_df = seed_df[seed_df["category"].isin(allowed_categories)].reset_index(drop=True)
    
    # generate synth rows with source-aware counts
    synth_rows = []
    # 1) Admin/user feedback-based synth
    for _, row in accepted.iterrows():
        source = row.get('source', 'user')
        merchant = extract_merchant_token(row['transaction_text']) or 'merchant'
        amt_match = re.search(r"(\d+[.,]?\d*)", row['transaction_text'])
        amt = float(amt_match.group(1).replace(',', '')) if amt_match is not None else None
        n = SYNTH_ADMIN if source == 'admin' else SYNTH_USER
        variants = generate_variants(merchant, amount=amt, n=n)
        for v in variants:
            synth_rows.append({'transaction_text': v, 'label': row['true_category'], 'row_source': source})

    # 2) Admin-synonym seeds → always treated as admin-weighted
    if not seed_df.empty:
        for _, r in seed_df.iterrows():
            merchant = r["merchant"]
            cat = r["category"]
            variants = generate_variants(merchant, amount=None, n=SYNTH_ADMIN_SYNONYM)
            for v in variants:
                synth_rows.append({"transaction_text": v, "label": cat, "row_source": "admin"})
        # mark consumed
        mark_seeds_used(set(seed_df["merchant"]), set(seed_df["category"]))
        print(f"🧩 Consumed {len(seed_df)} admin synonym seed(s) → generated {len(seed_df)*SYNTH_ADMIN_SYNONYM} rows.")

    synth_df = pd.DataFrame(synth_rows)
    print(f"🔧 Generated {len(synth_df)} synthetic rows (by source):", synth_df['row_source'].value_counts().to_dict())

    # merge with main
    base = main_df[['transaction_text','label']].copy()
    base['row_source'] = 'main'
    train_df = pd.concat([base, synth_df], ignore_index=True)
    train_df['label'] = train_df['label'].astype(str).str.strip().str.lower()
    train_df = train_df[train_df['label'] != 'nan']
    train_df = train_df.drop_duplicates(subset=['transaction_text','label'], keep='last').reset_index(drop=True)

    print(f"📊 Training set size after augmentation: {len(train_df)} (classes: {train_df['label'].nunique()})")
    print(train_df['label'].value_counts().head(20))

    # build sample weights
    def src_weight(src):
        if src == 'main':
            return WEIGHT_MAIN
        elif src == 'admin':
            return WEIGHT_ADMIN
        else:
            return WEIGHT_USER

    weights = train_df['row_source'].apply(src_weight).values

    # safe split (split arrays in same order)
    label_counts = train_df['label'].value_counts()
    stratify_param = train_df['label'] if int(label_counts.min()) >= 2 else None
    if stratify_param is None:
        print(f"⚠️ Some classes have <2 samples (min={int(label_counts.min())}). Using non-stratified split.")

    X = train_df['transaction_text'].values
    y = train_df['label'].values
    w = weights

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify_param
    )

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    model = LogisticRegression(max_iter=400, class_weight='balanced')
    model.fit(X_train_vec, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_vec)
    print(f"\n🏁 Macro F1 after retrain: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    print(f"💾 Saved model -> {MODEL_PATH}")
    print(f"💾 Saved vectorizer -> {VECT_PATH}")

    # remove accepted rows from FEEDBACK, keep pending
    remaining = pending.copy()
    remaining.to_csv(FEEDBACK, index=False)
    print("🧹 Processed accepted feedback removed from user_feedback.csv; pending rows retained for admin review.")

if __name__ == "__main__":
    main()
