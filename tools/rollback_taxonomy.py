# tools/rollback_taxonomy.py
from tools.taxonomy import atomic_write_yaml, backup_taxonomy
import shutil

def rollback(backup_path, user="admin"):
    obj = yaml.safe_load(open(backup_path))
    atomic_write_yaml(obj)
    append_audit(user=user, action="rollback", summary=f"Rollback to {backup_path}", before_path=None, after_obj=obj)
