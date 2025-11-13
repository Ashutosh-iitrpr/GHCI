# tools/taxonomy_admin.py
import sys
import os
import subprocess
import yaml #type: ignore
import pandas as pd #type: ignore
import streamlit as st  #type: ignore
import time

# ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.taxonomy import (
    load_taxonomy,
    update_taxonomy,
    promote_pending_category,
    add_category,
    add_merchant,
)

st.set_page_config(page_title="Taxonomy Admin (debuggable)", layout="wide")
st.title("Taxonomy Admin (debuggable)")

# -------------------------
# Debug/log helpers
# -------------------------
DEBUG_LOG = os.path.join(PROJECT_ROOT, "data", "admin_debug.log")
RETRAIN_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RETRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "data", "retrain_with_feedback.py")
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)

def write_debug(msg: str):
    ts = pd.Timestamp.now().isoformat()
    line = f"[{ts}] {msg}"
    # print to console (visible where streamlit was started)
    print(line, flush=True)
    # append to debug log file
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[debug-write-failed] {e}", flush=True)

def run_retrain_and_confirm(timeout_seconds: int = 8):
    """
    Launch retrain script located at PROJECT_ROOT/data/retrain_with_feedback.py.
    Write stdout/stderr to logs/retrain_stdout.log (utf-8).
    Poll the logfile for expected markers for up to timeout_seconds to confirm the job started.
    Returns (success:bool, detail:str).
    """
    write_debug("run_retrain_and_confirm() called")
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
        write_debug(f"Retrain subprocess launched: pid={proc.pid}, script={RETRAIN_SCRIPT}, cwd={PROJECT_ROOT}, log={retrain_log_path}")
    except Exception as e:
        msg = f"Failed to launch retrain subprocess: {e}"
        write_debug(msg)
        return False, msg

    # poll logfile for evidence (short window)
    evidence_tokens = [
        "Loading main dataset", "Loading main dataset...", "Loading main dataset…",
        "Generated", "Consummed", "Consumed", "Saved model", "Saved model ->",
        "🧩", "🔧", "Error", "Traceback"
    ]
    start = time.time()
    seen = False
    last_tail = ""
    while time.time() - start < timeout_seconds:
        time.sleep(0.3)
        try:
            with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                content = rf.read()
            # short-circuit if obvious evidence present
            for tok in evidence_tokens:
                if tok in content:
                    seen = True
                    break
            if seen:
                write_debug(f"Evidence token found in retrain log within {time.time()-start:.2f}s")
                return True, f"Retrain launched (pid={proc.pid}), evidence seen in log."
        except Exception as e:
            # file might be locked momentarily; ignore and continue
            write_debug(f"Error reading retrain log while polling: {e}")
            continue

        # also check if process died quickly
        if proc.poll() is not None:
            # process finished earlier than timeout: capture tail and return failure
            try:
                with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                    tail = rf.read()[-4000:]
            except Exception:
                tail = "<could not read log tail>"
            msg = f"Retrain process exited quickly (returncode={proc.returncode}). Log tail:\n{tail}"
            write_debug(msg)
            return False, msg

    # timed out without explicit evidence but process still alive: treat as launched
    if proc.poll() is None:
        msg = f"Retrain launched (pid={proc.pid}) but no evidence token seen within {timeout_seconds}s. Check logs/retrain_stdout.log for progress."
        write_debug(msg)
        return True, msg
    else:
        # process ended but no evidence found
        try:
            with open(retrain_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                tail = rf.read()[-4000:]
        except Exception:
            tail = "<could not read log tail>"
        msg = f"Retrain finished (returncode={proc.returncode}) but no expected output. Log tail:\n{tail}"
        write_debug(msg)
        return False, msg

# -------------------------
# Small helpers for data files
# -------------------------
def get_categories():
    tax = load_taxonomy()
    return tax, sorted(list(tax.get("categories", {}).keys()))

def read_pending_df(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "pending_new_categories.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna(how="all", axis=1)
    if "true_category" not in df.columns:
        df["true_category"] = ""
    df["true_category"] = df["true_category"].astype(str).str.strip().str.lower()
    return df

def read_suggestions_df(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "merchant_suggestions.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna(how="all", axis=1)
    if "merchant" not in df.columns:
        df["merchant"] = ""
    if "suggested_category" not in df.columns:
        df["suggested_category"] = ""
    df["merchant"] = df["merchant"].astype(str).str.strip().str.lower()
    df["suggested_category"] = df["suggested_category"].astype(str).str.strip().str.lower()
    df = df[df["merchant"] != ""].drop_duplicates().reset_index(drop=True)
    return df

def save_suggestions_df(df, path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "merchant_suggestions.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def remove_suggestions(merchants, path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "merchant_suggestions.csv")
    if not os.path.exists(path):
        return 0
    df = read_suggestions_df(path)
    if df is None or df.empty:
        return 0
    keep = df[~df["merchant"].isin(set(merchants))]
    save_suggestions_df(keep, path)
    return len(df) - len(keep)

# -------------------------
# UI sections
# -------------------------
tax, categories = get_categories()
st.subheader(f"Existing categories ({len(categories)})")
st.write(categories)

st.markdown("---")
st.subheader("Upload new taxonomy YAML")
uploaded = st.file_uploader("Upload taxonomy YAML file", type=["yaml", "yml"])
if uploaded:
    try:
        new_obj = yaml.safe_load(uploaded.read())
        st.write("Preview categories:", list(new_obj.get("categories", {}).keys()))
        if st.button("Validate & update taxonomy"):
            write_debug("Upload taxonomy button clicked")
            try:
                update_taxonomy(new_obj, user="streamlit-admin", trigger_retrain=False)
                write_debug("update_taxonomy() completed for uploaded taxonomy")
                # automatically retrain after upload
                launched = run_retrain_and_confirm()
                write_debug(f"run_retrain returned {launched} after taxonomy upload")
            except Exception as e:
                write_debug(f"update_taxonomy() failed: {e}")
                st.error(f"Validation/write failed: {e}")
    except Exception as e:
        st.error(f"Could not parse uploaded YAML: {e}")

st.markdown("---")
st.subheader("Add a new category (admin)")
with st.form("add_category_form"):
    new_key = st.text_input("Category key (slug, lowercase)", help="e.g. donations")
    display_name = st.text_input("Display name (optional)", value="")
    synonyms_txt = st.text_input("Initial merchants (comma-separated, optional)", value="")
    description = st.text_area("Description (optional)", value="")
    submitted = st.form_submit_button("Create category")
    if submitted:
        write_debug(f"Add category form submitted: key={new_key}, display_name={display_name}, synonyms={synonyms_txt}")
        try:
            syn_list = [s.strip().lower() for s in synonyms_txt.split(",") if s.strip()]
            add_category(new_key, display_name=display_name or None, synonyms=syn_list, description=description, user="streamlit-admin", trigger_retrain=False)
            write_debug(f"add_category() succeeded for '{new_key}', syns={syn_list}")
            # trigger retrain and log
            launched = run_retrain_and_confirm()
            write_debug(f"run_retrain returned {launched} after add_category")
            try:
                # attempt automatic page refresh
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    raise AttributeError("st.experimental_rerun not present")
            except Exception as e:
                write_debug(f"st.experimental_rerun not available or failed: {e}")
                # provide user-friendly fallback
                st.info("Please manually refresh the page to reflect the changes (browser refresh).")

        except Exception as e:
            write_debug(f"add_category() failed: {e}")
            st.error(f"Failed to add category: {e}")

st.markdown("---")
st.subheader("Add a merchant to an existing category (admin)")
with st.form("add_merchant_form"):
    _, categories = get_categories()
    if not categories:
        st.info("No categories available. Create one first.")
        submitted2 = st.form_submit_button("Add merchant", disabled=True)
    else:
        sel_cat = st.selectbox("Select category", categories)
        new_merchant = st.text_input("Merchant token to add (e.g. nykaa, amazon)")
        submitted2 = st.form_submit_button("Add merchant")
        if submitted2:
            write_debug(f"Add merchant form submitted: merchant={new_merchant}, category={sel_cat}")
            if not new_merchant.strip():
                st.error("Provide merchant name.")
            else:
                try:
                    ok = add_merchant(sel_cat, new_merchant.strip().lower(), user="streamlit-admin", trigger_retrain=False)
                    if ok:
                        removed = remove_suggestions([new_merchant.strip().lower()])
                        write_debug(f"add_merchant() succeeded: {new_merchant} -> {sel_cat}; removed_suggestions={removed}")
                        # trigger retrain and log
                        launched = run_retrain_and_confirm()
                        write_debug(f"run_retrain returned {launched} after add_merchant")
                        st.success(f"Added merchant '{new_merchant}' to category '{sel_cat}'. Removed from suggestions: {removed}")
                    else:
                        write_debug(f"add_merchant() returned False (already exists): {new_merchant} -> {sel_cat}")
                        st.info("Merchant already exists in this category's synonyms.")
                    try:
                        # attempt automatic page refresh
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                        else:
                            raise AttributeError("st.experimental_rerun not present")
                    except Exception as e:
                        write_debug(f"st.experimental_rerun not available or failed: {e}")
                        # provide user-friendly fallback
                        st.info("Please manually refresh the page to reflect the changes (browser refresh).")

                except Exception as e:
                    write_debug(f"add_merchant() failed: {e}")
                    st.error(f"Failed to add merchant: {e}")

st.markdown("---")
st.subheader("Pending new categories (user feedback)")
pending_df = read_pending_df()
if pending_df is None or pending_df.empty:
    st.info("No pending_new_categories.csv found (no unaccepted user categories).")
else:
    pending_counts = pending_df["true_category"].value_counts()
    st.write("Pending counts by category (user-provided):")
    st.dataframe(pending_counts.reset_index().rename(columns={"index": "category", "true_category": "count"}), width="stretch")

    st.markdown("Promote a pending category (moves rows to user_feedback.csv and tags them as admin):")
    pending_cats = list(pending_counts.index)
    if pending_cats:
        sel = st.selectbox("Pending category to promote", pending_cats, key="sel_pending_cat")
        if st.button("Promote selected pending category and mark as admin"):
            write_debug(f"Promote button clicked for category: {sel}")
            try:
                promoted = promote_pending_category(sel)
                write_debug(f"promote_pending_category() returned {promoted} for {sel}")
                # trigger retrain and log
                launched = run_retrain_and_confirm()
                write_debug(f"run_retrain returned {launched} after promote_pending_category")
                st.success(f"Promoted {promoted} rows for category '{sel}' as admin-sourced into user_feedback.csv")
                try:
                    # attempt automatic page refresh
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        raise AttributeError("st.experimental_rerun not present")
                except Exception as e:
                    write_debug(f"st.experimental_rerun not available or failed: {e}")
                    # provide user-friendly fallback
                    st.info("Please manually refresh the page to reflect the changes (browser refresh).")

            except Exception as e:
                write_debug(f"promote_pending_category() failed: {e}")
                st.error(f"Promotion failed: {e}")
    else:
        st.info("No pending categories to promote.")

st.markdown("---")
st.subheader("Merchant suggestions (from accepted user feedback)")
suggest_df = read_suggestions_df()
if suggest_df is None or suggest_df.empty:
    st.info("No merchant_suggestions.csv found yet.")
else:
    st.write("These merchants appeared in accepted feedback but are not in taxonomy synonyms yet.")
    st.dataframe(suggest_df, width="stretch")

    # single merchant add with chosen category
    st.markdown("### Add ONE suggested merchant with chosen category")
    merch_options = list(suggest_df["merchant"].unique())
    sel_one = st.selectbox("Merchant", merch_options, key="sel_one_merchant")
    default_cat = ""
    try:
        row = suggest_df[suggest_df["merchant"] == sel_one].iloc[0]
        default_cat = str(row.get("suggested_category", "") or "")
    except Exception:
        default_cat = ""
    _, categories = get_categories()
    if default_cat not in categories and categories:
        default_cat = categories[0]
    sel_cat_one = st.selectbox("Category for this merchant", categories, index=(categories.index(default_cat) if default_cat in categories else 0), key="sel_one_merchant_cat")
    
    if st.button("Add merchant to chosen category"):
        write_debug(f"Add ONE suggested merchant button clicked: merchant={sel_one}, target_cat={sel_cat_one}")
        try:
            ok = add_merchant(sel_cat_one, sel_one, user="streamlit-admin", trigger_retrain=False)
            if ok:
                removed = remove_suggestions([sel_one])
                write_debug(f"add_merchant() (from suggestion) succeeded: {sel_one} -> {sel_cat_one}; removed={removed}")
                # robust retrain call
                success, detail = run_retrain_and_confirm(timeout_seconds=8)
                write_debug(f"run_retrain_and_confirm() returned: success={success}, detail={detail}")
                if success:
                    st.success(f"Added '{sel_one}' → '{sel_cat_one}'. Retrain started. {detail}")
                else:
                    st.error(f"Added '{sel_one}' → '{sel_cat_one}', but retrain failed to start: {detail}")
            else:
                write_debug(f"add_merchant() returned False (already exists): {sel_one} -> {sel_cat_one}")
                st.info("Merchant already exists in taxonomy synonyms.")
        except Exception as e:
            write_debug(f"add_merchant() (from suggestion) failed: {e}")
            st.error(f"Failed to add merchant: {e}")

    st.markdown("---")
    # bulk add to a selected category
    st.markdown("**Bulk add selected merchants to a category**")
    merch_options = list(suggest_df["merchant"].unique())
    selected_merchants = st.multiselect("Pick merchants to add", merch_options, key="ms_merchants_bulk")
    target_cat = st.selectbox("Target category", categories, key="sel_bulk_target_cat")
    if st.button("Add selected merchants to target category"):
        write_debug(f"Bulk add button clicked: merchants={selected_merchants}, target={target_cat}")
        if not selected_merchants:
            st.warning("Pick at least one merchant.")
        else:
            added = 0
            for m in selected_merchants:
                try:
                    ok = add_merchant(target_cat, m, user="streamlit-admin", trigger_retrain=False)
                    if ok:
                        added += 1
                        write_debug(f"add_merchant() bulk: added {m} -> {target_cat}")
                    else:
                        write_debug(f"add_merchant() bulk: skipped existing {m} -> {target_cat}")
                except Exception as e:
                    write_debug(f"add_merchant() bulk failed for {m}: {e}")
                    st.error(f"Failed to add '{m}': {e}")
            removed = remove_suggestions(selected_merchants)
            write_debug(f"Bulk add finished: added={added}, removed_suggestions={removed}")
            if added > 0:
                launched = run_retrain_and_confirm()
                write_debug(f"run_retrain returned {launched} after bulk add")
            try:
                # attempt automatic page refresh
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    raise AttributeError("st.experimental_rerun not present")
            except Exception as e:
                write_debug(f"st.experimental_rerun not available or failed: {e}")
                # provide user-friendly fallback
                st.info("Please manually refresh the page to reflect the changes (browser refresh).")


    st.markdown("---")
    # bulk add using suggested category
    st.markdown("**Bulk add using each merchant's suggested category**")
    unique_rows = suggest_df.drop_duplicates(subset=["merchant", "suggested_category"]).copy()
    valid_rows = unique_rows[unique_rows["suggested_category"].isin(categories)]
    if valid_rows.empty:
        st.info("No rows with valid suggested categories found.")
    else:
        display_opts = [f"{r['merchant']} → {r['suggested_category']}" for _, r in valid_rows.iterrows()]
        selected_rows = st.multiselect("Pick merchants to add with their suggested category", options=display_opts, key="ms_merchants_suggested")
        if st.button("Add selected merchants using suggested categories"):
            write_debug(f"Bulk-add-by-suggested clicked: selections={selected_rows}")
            if not selected_rows:
                st.warning("Pick at least one merchant.")
            else:
                to_add = []
                for s in selected_rows:
                    merchant, _, cat = s.partition(" → ")
                    to_add.append((merchant.strip(), cat.strip()))
                added = 0
                for m, c in to_add:
                    try:
                        ok = add_merchant(c, m, user="streamlit-admin", trigger_retrain=False)
                        if ok:
                            added += 1
                            write_debug(f"add_merchant() suggested: added {m} -> {c}")
                        else:
                            write_debug(f"add_merchant() suggested: skipped existing {m} -> {c}")
                    except Exception as e:
                        write_debug(f"add_merchant() suggested failed for {m}->{c}: {e}")
                        st.error(f"Failed to add '{m}' → '{c}': {e}")
                removed = remove_suggestions([m for m, _ in to_add])
                write_debug(f"Bulk-by-suggested finished: added={added}, removed={removed}")
                if added > 0:
                    launched = run_retrain_and_confirm()
                    write_debug(f"run_retrain returned {launched} after bulk suggested add")
                try:
                    # attempt automatic page refresh
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        raise AttributeError("st.experimental_rerun not present")
                except Exception as e:
                    write_debug(f"st.experimental_rerun not available or failed: {e}")
                    # provide user-friendly fallback
                    st.info("Please manually refresh the page to reflect the changes (browser refresh).")


st.markdown("---")
st.subheader("Utilities / Debug")
if st.button("Show admin debug logfile path"):
    st.write(os.path.abspath(DEBUG_LOG))
if st.button("Show retrain stdout log path"):
    st.write(os.path.abspath(os.path.join(RETRAIN_LOG_DIR, "retrain_stdout.log")))
