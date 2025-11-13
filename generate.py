import random
import pandas as pd
import yaml
from datetime import datetime, timedelta
import string
import os

# ----------------------
# Utility functions
# ----------------------

def introduce_typo(s: str, prob: float = 0.1) -> str:
    """Randomly introduces typos in ~prob% of characters."""
    s_list = list(s)
    for i in range(len(s_list)):
        if random.random() < prob:
            s_list[i] = random.choice(string.ascii_lowercase)
    return "".join(s_list)

def random_case(s: str):
    """Randomly upper/lower case a string."""
    mode = random.choice(["lower", "upper", "title", "mixed"])
    if mode == "lower": return s.lower()
    if mode == "upper": return s.upper()
    if mode == "title": return s.title()
    if mode == "mixed":
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in s)
    return s

def random_txn_pattern(merchant: str, amount: float):
    """Generates random transaction text patterns."""
    templates = [
        f"POS {merchant} {int(amount)} TXN",
        f"UPI payment to {merchant}@ybl",
        f"PURCHASE {merchant} Ref#{random.randint(1000,9999)}",
        f"Online payment to {merchant}",
        f"DEBIT CARD {merchant.upper()} {int(amount)}",
        f"PAYMENT TO {merchant} VIA NETBANKING",
        f"Txn {random.randint(10000,99999)}: {merchant} {amount}",
        f"IMPS transfer to {merchant}",
        f"{merchant} *{random.randint(1000,9999)} POS Txn",
    ]
    return random.choice(templates)

def add_random_tokens(s: str):
    """Adds random finance-like tokens to increase realism."""
    tokens = ["TXN", "Ref", "POS", "IMPS", "UPI", "NEFT", "Online", "PAY"]
    if random.random() < 0.5:
        s = s + " " + random.choice(tokens) + str(random.randint(10, 9999))
    return s

# ----------------------
# Main generation logic
# ----------------------

def generate_transactions(taxonomy_path: str, n_per_category: int = 1000, noise_level: float = 0.15):
    """Generates synthetic transactions dataset."""
    
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.safe_load(f)['categories']
    
    data = []
    for category, merchants in taxonomy.items():
        for _ in range(n_per_category):
            merchant = random.choice(merchants)
            amount = round(random.uniform(50, 5000), 2)
            days_ago = random.randint(1, 365)
            date = datetime.now() - timedelta(days=days_ago)
            
            txn_text = random_txn_pattern(merchant, amount)
            
            # Add controlled noise
            if random.random() < noise_level:
                txn_text = introduce_typo(txn_text, prob=0.05)
            if random.random() < noise_level:
                txn_text = add_random_tokens(txn_text)
            if random.random() < noise_level:
                txn_text = random_case(txn_text)
            
            data.append({
                "transaction_text": txn_text,
                "merchant": merchant,
                "amount": amount,
                "date": date.strftime("%Y-%m-%d"),
                "label": category
            })
    
    df = pd.DataFrame(data)
    return df

# ----------------------
# Script entrypoint
# ----------------------

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    df = generate_transactions(
        taxonomy_path="config/taxonomy.yaml",
        n_per_category=1200,     # 1200 per category → ~7k total
        noise_level=0.2          # 20% noisy samples
    )
    
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    out_path = "data/synthetic_transactions.csv"
    df.to_csv(out_path, index=False)
    
    print(f"✅ Generated {len(df)} transactions and saved to {out_path}")
    print(df.sample(10))
