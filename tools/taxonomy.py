# tools/taxonomy.py
import os
import yaml #type: ignore
import json
import shutil
import tempfile
import datetime
from jsonschema import validate, ValidationError #type: ignore
import pandas as pd #type: ignore
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAX_PATH = "config/taxonomy.yaml"
BACKUP_DIR = "config/backups"
AUDIT_LOG = "data/taxonomy_changes.log"

SCHEMA = {
    "type": "object",
    "required": ["categories"],
    "properties": {
        "version": {"type": "integer"},
        "updated_by": {"type": "string"},
        "updated_at": {"type": "string"},
        "categories": {
            "type": "object",
            "patternProperties": {
                "^[a-z0-9_\\- ]+$": {
                    "type": "object",
                    "required": ["display_name"],
                    "properties": {
                        "display_name": {"type": "string"},
                        "synonyms": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            }
        }
    }
}

def load_taxonomy(path=TAX_PATH):
    if not os.path.exists(path):
        return {"version": 1, "updated_by": "none", "updated_at": datetime.datetime.utcnow().isoformat(), "categories": {}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"version": 1, "categories": {}}

def validate_taxonomy(obj):
    validate(instance=obj, schema=SCHEMA)

def backup_taxonomy(path=TAX_PATH, backup_dir=BACKUP_DIR):
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = os.path.join(backup_dir, f"taxonomy.{ts}.yaml")
    shutil.copy2(path, dst)
    return dst

def atomic_write_yaml(obj, path=TAX_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="taxonomy_", dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True)
    os.replace(tmp, path)

def append_audit(user, action, summary, before_path=None, after_obj=None, audit_log=AUDIT_LOG):
    os.makedirs(os.path.dirname(audit_log), exist_ok=True)
    ts = datetime.datetime.utcnow().isoformat()
    entry = {
        "timestamp": ts,
        "user": user,
        "action": action,
        "summary": summary
    }
    if before_path:
        entry["before_backup"] = before_path
    if after_obj:
        entry["after_snapshot"] = {"version": after_obj.get("version"), "categories": list(after_obj.get("categories", {}).keys())}
    with open(audit_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def update_taxonomy(new_obj, user="admin", trigger_retrain=False, retrain_cmd=None):
    # validate
    try:
        validate_taxonomy(new_obj)
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e.message}")

    # backup
    before_backup = None
    if os.path.exists(TAX_PATH):
        before_backup = backup_taxonomy()

    # fill metadata
    new_obj = dict(new_obj)  # shallow copy
    new_obj.setdefault("version", 1)
    if os.path.exists(TAX_PATH):
        try:
            prev = load_taxonomy()
            new_obj["version"] = int(prev.get("version", 1)) + 1
        except Exception:
            new_obj["version"] = new_obj.get("version", 1)
    new_obj["updated_by"] = user
    new_obj["updated_at"] = datetime.datetime.utcnow().isoformat()

    # atomic write
    atomic_write_yaml(new_obj)

    # audit
    append_audit(user=user, action="update", summary=f"Updated taxonomy ({len(new_obj.get('categories', {}))} categories)", before_path=before_backup, after_obj=new_obj)

    # optional retrain trigger (non-blocking)
    if trigger_retrain and retrain_cmd:
        import subprocess
        subprocess.Popen(retrain_cmd, shell=True)

    return True

# -------------------------
# Pending promotion helper
# -------------------------
def promote_pending_category(category_name):
    """
    Move rows in data/pending_new_categories.csv whose true_category == category_name
    into data/user_feedback.csv and tag them source=admin. Returns number moved.
    """
    pending_path = "data/pending_new_categories.csv"
    feedback_path = "data/user_feedback.csv"
    if not os.path.exists(pending_path):
        return 0

    prev = pd.read_csv(pending_path).dropna(how="all", axis=1)
    if "true_category" not in prev.columns:
        prev["true_category"] = ""
    prev["true_category"] = prev["true_category"].astype(str).str.strip().str.lower()

    to_promote = prev[prev["true_category"] == category_name].copy()
    if to_promote.empty:
        return 0

    # tag them as admin-sourced
    to_promote["source"] = "admin"

    # append to feedback (create if not exists)
    if os.path.exists(feedback_path):
        fb = pd.read_csv(feedback_path).dropna(how="all", axis=1)
        new_fb = pd.concat([fb, to_promote], ignore_index=True).drop_duplicates(subset=["transaction_text","true_category"], keep="last")
    else:
        new_fb = to_promote

    os.makedirs(os.path.dirname(feedback_path) or ".", exist_ok=True)
    new_fb.to_csv(feedback_path, index=False)

    # remove promoted rows from pending and write back
    remaining = prev[prev["true_category"] != category_name]
    remaining.to_csv(pending_path, index=False)

    append_audit(user="admin", action="promote", summary=f"Promoted {len(to_promote)} pending rows for category '{category_name}'", after_obj={"categories": {category_name: {}}})
    return len(to_promote)

# -------------------------
# Admin helpers: add category / merchant
# -------------------------
def add_category(category_key, display_name=None, synonyms=None, description="", user="admin", trigger_retrain=False, retrain_cmd=None):
    """
    Add a new category to taxonomy. category_key should be a lowercase slug-like string.
    synonyms: optional list of merchant tokens (strings).
    """
    category_key = str(category_key).strip().lower()
    if not category_key:
        raise ValueError("category_key empty")

    tax = load_taxonomy()
    cats = tax.get("categories", {})
    if category_key in cats:
        raise ValueError(f"Category '{category_key}' already exists")

    entry = {
        "display_name": display_name or category_key.title(),
        "synonyms": list(synonyms) if synonyms else [],
        "description": description or ""
    }
    cats[category_key] = entry
    tax["categories"] = cats
    # update metadata and save via update_taxonomy (validation + backup + audit)
    update_taxonomy(tax, user=user, trigger_retrain=trigger_retrain, retrain_cmd=retrain_cmd)
    return True

def _append_synonym_seed(category_key, merchant_name, path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "taxonomy_synonym_seeds.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"merchant": merchant_name.strip().lower(), "category": category_key.strip().lower(), "used": 0}
    if os.path.exists(path):
        df = pd.read_csv(path).dropna(how="all", axis=1)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)\
               .drop_duplicates(subset=["merchant", "category"], keep="last")
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


def add_merchant(category_key, merchant_name, user="admin", trigger_retrain=False, retrain_cmd=None):
    """
    Add a merchant token into an existing category's synonyms list.
    """
    category_key = str(category_key).strip().lower()
    merchant_name = str(merchant_name).strip().lower()
    if not category_key or not merchant_name:
        raise ValueError("category or merchant empty")

    tax = load_taxonomy()
    cats = tax.get("categories", {})
    if category_key not in cats:
        raise ValueError(f"Category '{category_key}' does not exist")

    syn = cats[category_key].get("synonyms", [])
    if merchant_name in syn:
        return False  # already exists

    syn.append(merchant_name)
    cats[category_key]["synonyms"] = syn
    tax["categories"] = cats
    update_taxonomy(tax, user=user, trigger_retrain=trigger_retrain, retrain_cmd=retrain_cmd)
    _append_synonym_seed(category_key, merchant_name)
    append_audit(user=user, action="add_merchant", summary=f"Added merchant '{merchant_name}' to category '{category_key}'", after_obj={"categories": {category_key: cats[category_key]}})
    return True

