# tools/manage_taxonomy.py
import argparse
import yaml #type: ignore
from tools.taxonomy import update_taxonomy

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="YAML file with new taxonomy")
parser.add_argument("--user", default="admin", help="admin user id/email")
parser.add_argument("--retrain", action="store_true", help="trigger retrain after update")
args = parser.parse_args()

with open(args.file, "r", encoding="utf-8") as f:
    new_tax = yaml.safe_load(f)

retrain_cmd = "python retrain_with_feedback.py" if args.retrain else None
update_taxonomy(new_tax, user=args.user, trigger_retrain=args.retrain, retrain_cmd=retrain_cmd)
print("✅ taxonomy updated")