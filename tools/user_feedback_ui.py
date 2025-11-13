# tools/user_feedback_ui.py
import os
import sys
import time
import subprocess
import pandas as pd #type: ignore
import yaml #type: ignore
import streamlit as st #type: ignore

# -------------------------
# Project root + paths
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEEDBACK_CSV = os.path.join(DATA_DIR, "user_feedback.csv")  # predictions written here by predict_transaction.py
DEBUG_LOG = os.path.join(DATA_DIR, "user_feedback_debug.log")

RETRAIN_SCRIPT = os.path.join(DATA_DIR, "retrain_with_feedback.py")
RETRAIN_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)

TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "config", "taxonomy.yaml")

# -------------------------
# Helpers
# -------------------------
def write_debug(msg: str):
    ts = pd.Timestamp.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_feedback_df():
    if not os.path.exists(FEEDBACK_CSV):
        # create empty DF with expected columns if not present
        cols = ["transaction_text", "predicted_category", "confidence", "review_required", "true_category", "source", "timestamp"]
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(FEEDBACK_CSV).dropna(how="all", axis=1)
    except Exception:
        # if file malformed, try to read with engine python
        df = pd.read_csv(FEEDBACK_CSV, engine="python", error_bad_lines=False)
    # ensure expected cols exist
    for c in ["transaction_text","predicted_category","confidence","review_required","true_category","source","timestamp"]:
        if c not in df.columns:
            df[c] = ""
    return df

def load_taxonomy_categories():
    if not os.path.exists(TAXONOMY_PATH):
        return []
    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            tax = yaml.safe_load(f) or {}
        cats = list(tax.get("categories", {}).keys())
        return sorted([str(c).strip() for c in cats if str(c).strip()])
    except Exception as e:
        write_debug(f"Failed to load taxonomy: {e}")
        return []

def append_or_update_feedback_rows(df_original, df_updates):
    """
    Merge df_updates back into df_original by index (if index preserved) or by transaction_text.
    We set source='user' and timestamp for updated rows.
    Save back to FEEDBACK_CSV with UTF-8 encoding.
    Returns number of rows updated.
    """
    updated = 0
    # Ensure index alignment: if df_updates has an '_orig_index' column, use it
    if "_orig_index" in df_updates.columns:
        for _, r in df_updates.iterrows():
            try:
                idx = int(r["_orig_index"])
            except Exception:
                continue
            if idx in df_original.index:
                # update true_category if not empty
                new_label = str(r.get("true_category", "")).strip()
                if new_label:
                    df_original.at[idx, "true_category"] = new_label
                    df_original.at[idx, "source"] = "user"
                    df_original.at[idx, "timestamp"] = pd.Timestamp.now().isoformat()
                    updated += 1
    else:
        # fallback: match by transaction_text
        for _, r in df_updates.iterrows():
            tx = str(r.get("transaction_text","")).strip()
            new_label = str(r.get("true_category", "")).strip()
            if not tx or not new_label:
                continue
            matches = df_original[df_original["transaction_text"].astype(str).str.strip() == tx].index.tolist()
            if not matches:
                continue
            for idx in matches:
                df_original.at[idx, "true_category"] = new_label
                df_original.at[idx, "source"] = "user"
                df_original.at[idx, "timestamp"] = pd.Timestamp.now().isoformat()
                updated += 1
    # save
    os.makedirs(os.path.dirname(FEEDBACK_CSV) or ".", exist_ok=True)
    df_original.to_csv(FEEDBACK_CSV, index=False, encoding="utf-8")
    return updated

def run_retrain_and_confirm(timeout_seconds: int = 8):
    """
    Launch retrain script located at PROJECT_ROOT/data/retrain_with_feedback.py.
    Write stdout/stderr to logs/retrain_stdout.log (utf-8).
    Poll the logfile for expected markers for up to timeout_seconds to confirm the job started.
    Returns (success:bool, detail:str).
    """
    write_debug("run_retrain_and_confirm() called (user UI)")

    if not os.path.exists(RETRAIN_SCRIPT):
        msg = f"RETRAIN SCRIPT NOT FOUND: {RETRAIN_SCRIPT}"
        write_debug(msg)
        return False, msg

    retrain_log_path = os.path.join(RETRAIN_LOG_DIR, "retrain_stdout.log")
    try:
        logf = open(retrain_log_path, "a", buffering=1, encoding="utf-8")
    except Exception as e:
        msg = f"Failed to open retrain log file: {e}"
        write_debug(msg)
        return False, msg

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen([sys.executable, RETRAIN_SCRIPT],
                                cwd=PROJECT_ROOT,
                                stdout=logf,
                                stderr=logf,
                                env=env,
                                shell=False)
        write_debug(f"Retrain subprocess launched (user UI): pid={proc.pid}, script={RETRAIN_SCRIPT}, cwd={PROJECT_ROOT}, log={retrain_log_path}")
    except Exception as e:
        msg = f"Failed to launch retrain subprocess: {e}"
        write_debug(msg)
        return False, msg

    evidence_tokens = [
        "Loading main dataset", "Generated", "Consumed", "Saved model", "RETRAIN_START",
        "Error", "Traceback", "🧩", "🔧"
    ]
    start = time.time()
    seen = False
    while time.time() - start < timeout_seconds:
        time.sleep(0.3)
        try:
            with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                content = rf.read()
            for tok in evidence_tokens:
                if tok in content:
                    seen = True
                    break
            if seen:
                write_debug(f"Evidence token found in retrain log within {time.time()-start:.2f}s (user UI)")
                return True, f"Retrain launched (pid={proc.pid}), evidence seen in log."
        except Exception as e:
            write_debug(f"Error reading retrain log while polling: {e}")
            continue

        if proc.poll() is not None:
            try:
                with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                    tail = rf.read()[-4000:]
            except Exception:
                tail = "<could not read log tail>"
            msg = f"Retrain process exited quickly (returncode={proc.returncode}). Log tail:\n{tail}"
            write_debug(msg)
            return False, msg

    if proc.poll() is None:
        msg = f"Retrain launched (pid={proc.pid}) but no evidence token seen within {timeout_seconds}s. Check logs/retrain_stdout.log for progress."
        write_debug(msg)
        return True, msg
    else:
        try:
            with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                tail = rf.read()[-4000:]
        except Exception:
            tail = "<could not read log tail>"
        msg = f"Retrain finished (returncode={proc.returncode}) but no expected output. Log tail:\n{tail}"
        write_debug(msg)
        return False, msg

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="User Feedback Review", layout="wide")
st.title("User Feedback — review predicted transactions")

st.markdown(
    "This page shows the transactions that were predicted and saved in `data/user_feedback.csv`. "
    "Edit `true_category` inline for those rows you want to submit as user corrections, then click **Submit suggestions (user)**. "
    "Submissions will be recorded and will trigger a conservative retrain (user-sourced)."
)

# Load feedback df
df = load_feedback_df()
# add an index column to track rows during editing
df = df.reset_index(drop=False).rename(columns={"index":"_orig_index"}).set_index("_orig_index", drop=False)

# choose rows to show: default = rows needing review (true_category empty OR review_required contains Low)
needs_review_mask = (df["true_category"].astype(str).str.strip() == "") | (df["review_required"].astype(str).str.lower().str.contains("low"))
to_review = df[needs_review_mask].copy()

if to_review.empty:
    st.info("No predicted rows pending review (no rows with empty true_category or low-confidence flags).")
    st.markdown("You can still open `data/user_feedback.csv` to edit manually if needed.")
else:
    st.write(f"Showing {len(to_review)} rows pending review (editable).")

    # Provide taxonomy choices
    categories = load_taxonomy_categories()
    categories_display = ["<leave blank>"] + categories

    # Attempt to use Streamlit's data editor for inline editing
    edited = None
    used_data_editor = False
    try:
        # limit columns displayed to important ones and make true_category editable
        display_cols = ["transaction_text", "predicted_category", "confidence", "review_required", "true_category"]
        display_df = to_review[display_cols].copy()
        # preserve _orig_index so we can merge back
        display_df["_orig_index"] = to_review.index.astype(int)
        # st.data_editor (or st.experimental_data_editor) may exist depending on streamlit version
        if hasattr(st, "data_editor"):
            used_data_editor = True
            edited = st.data_editor(display_df, num_rows="dynamic")
        elif hasattr(st, "experimental_data_editor"):
            used_data_editor = True
            edited = st.experimental_data_editor(display_df, num_rows="dynamic")
        else:
            used_data_editor = False
            edited = None
    except Exception as e:
        write_debug(f"Data editor not available or failed: {e}")
        used_data_editor = False
        edited = None

    if not used_data_editor:
        # fallback: per-row manual editing (paginated for first 200 rows)
        st.warning("Inline table editor not available in this Streamlit version. Falling back to per-row editing UI.")
        edited_rows = []
        max_rows = min(200, len(to_review))
        for idx, row in to_review.iloc[:max_rows].iterrows():
            with st.expander(f"Row #{idx} — {row['transaction_text'][:80]}"):
                st.markdown(f"**Predicted:** {row.get('predicted_category','')}  —  **Confidence:** {row.get('confidence','')}")
                new_label = st.selectbox(f"Select true category for row {idx}", options=categories_display, key=f"sel_{idx}")
                if new_label == "<leave blank>":
                    new_label = ""
                edited_rows.append({
                    "_orig_index": int(idx),
                    "transaction_text": row["transaction_text"],
                    "true_category": new_label
                })
        if st.button("Submit suggestions (user)"):
            # build updates DF
            updates = pd.DataFrame(edited_rows)
            # apply updates
            updated_count = append_or_update_feedback_rows(df.reset_index(drop=True), updates)
            write_debug(f"User submitted {updated_count} rows (fallback editor).")
            # trigger retrain conservatively
            success, detail = run_retrain_and_confirm(timeout_seconds=8)
            if success:
                st.success(f"Saved {updated_count} suggestions and retrain started. {detail}")
            else:
                st.warning(f"Saved {updated_count} suggestions but retrain problem: {detail}")
    else:
        # data editor path: show a Submit button and merge edited rows back
        st.markdown("Edit the `true_category` column inline. When done, click Submit to save selected rows as user corrections.")
        st.caption("You can leave rows blank to skip them.")
        if edited is None:
            st.warning("Editor returned no data; refresh the page if this persists.")
        else:
            # edited contains interactive changes; ensure _orig_index preserved
            if "_orig_index" not in edited.columns:
                st.error("Editor did not preserve row index; aborting save.")
            else:
                # Display a checkbox to choose whether to only submit rows that changed
                submit_changed_only = st.checkbox("Submit only rows where `true_category` is non-empty", value=True)
                if st.button("Submit suggestions (user)"):
                    # build updates DataFrame
                    updates = edited[["_orig_index","transaction_text","true_category"]].copy()
                    # optionally filter to only rows where true_category non-empty
                    if submit_changed_only:
                        updates = updates[updates["true_category"].astype(str).str.strip() != ""]
                    if updates.empty:
                        st.info("No rows selected for submission. Nothing saved.")
                    else:
                        # convert index column to int
                        updates["_orig_index"] = updates["_orig_index"].astype(int)
                        # apply updates
                        updated_count = append_or_update_feedback_rows(df.reset_index(drop=True), updates)
                        write_debug(f"User submitted {updated_count} rows (data editor).")
                        # trigger retrain conservatively
                        success, detail = run_retrain_and_confirm(timeout_seconds=8)
                        if success:
                            st.success(f"Saved {updated_count} suggestions and retrain started. {detail}")
                        else:
                            st.warning(f"Saved {updated_count} suggestions but retrain problem: {detail}")

st.markdown("---")
st.write("Debug & log files:")
if st.button("Show debug logfile path"):
    st.write(os.path.abspath(DEBUG_LOG))
if st.button("Show retrain stdout log path"):
    st.write(os.path.abspath(os.path.join(RETRAIN_LOG_DIR, "retrain_stdout.log")))
