"""
data_cleaning.py - simple data cleaning for the NIDS project.
Works with NSL-KDD (KDDTrain+.txt / KDDTest+.txt) or a CSV with a header (e.g. RTNID_Dataset.csv).
Put the dataset in the project folder or in data/raw/, then run:  python data_cleaning.py
Output: data/processed/train_cleaned.csv, test_cleaned.csv, outputs/reports/cleaning_report.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level",
]
MISSING_TOKENS = {"nan", "null", "none", "na", "n/a"}
MAX_DUPLICATE_FRACTION = 0.05   # if more than 5% of rows are duplicates, report but do not remove


def clean_data():
    base = Path(__file__).resolve().parent
    search_dirs = [base / "data" / "raw", base]
    out_dir = base / "data" / "processed"
    report_path = base / "outputs" / "reports" / "cleaning_report.json"

    def find(name):
        return next((d / name for d in search_dirs if (d / name).is_file()), None)

    def is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    # ---- 1. Find files (NSL-KDD first, otherwise RTNID_Dataset.csv) ----
    files = {"train": find("KDDTrain+.txt"), "test": find("KDDTest+.txt")}
    if files["train"] is None:
        files = {"train": find("RTNID_Dataset.csv"), "test": None}
    if files["train"] is None:
        print("ERROR: dataset not found. Put KDDTrain+.txt / KDDTest+.txt (or RTNID_Dataset.csv) "
              f"in {base} or {base / 'data' / 'raw'}")
        return

    report = {}
    label_col, numeric_cols = None, None   # decided from the TRAIN file only

    for split, path in files.items():
        if path is None:
            continue
        print(f"\n=== {split.upper()}: {path} ===")

        # ---- 2. Load (everything as text so bad values stay visible) ----
        try:
            df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            print(f"ERROR: cannot read {path.name}: {error}")
            return
        first_row = [cell.strip() for cell in df.iloc[0]]
        if not any(is_number(cell) for cell in first_row):          # file has a header row
            names, seen = [], {}
            for name in first_row:                                   # make duplicate names unique
                seen[name] = seen.get(name, 0) + 1
                names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = names
        elif df.shape[1] == len(NSL_COLUMNS):                        # headerless NSL-KDD
            df.columns = NSL_COLUMNS
        else:
            print(f"ERROR: {path.name} has {df.shape[1]} columns and no header; cannot name columns.")
            return
        print("Shape:", df.shape)
        # print(df.head(5).to_string())

        # ---- 3. Column roles (from train only, reused for test) ----
        df = df.apply(lambda column: column.str.strip())             # remove whitespace
        if label_col is None:
            train_columns = list(df.columns)
            label_col = next((c for c in df.columns if c.lower() == "label"), df.columns[-1])
            numeric_cols = []
            for c in df.columns:
                if c == label_col:
                    continue
                present = df[c][df[c] != ""]
                if len(present) == 0 or pd.to_numeric(present, errors="coerce").notna().mean() >= 0.95:
                    numeric_cols.append(c)
        elif list(df.columns) != train_columns:
            print("ERROR: test columns do not match train columns.")
            return
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c != label_col]
        print("Label column:", label_col, "| categorical:", categorical_cols, "| numeric:", len(numeric_cols))

        # ---- 4. Inspect (before cleaning) ----
        original_rows = len(df)
        label_before = df[label_col].replace("", "<missing>").value_counts().to_dict()
        duplicates = int(df.duplicated().sum())
        print("\nData types:\n", df.dtypes.value_counts().to_string())
        print("Label distribution:", label_before)
        print("Duplicate rows:", duplicates)

        # ---- 5. Clean (never drops rows for missing/invalid/zero/extreme values) ----
        df = df.replace("", np.nan)
        invalid, infinite, placeholders = {}, {}, 0
        for c in numeric_cols:
            text = df[c]
            token_mask = text.str.lower().isin(MISSING_TOKENS)
            placeholders += int(token_mask.sum())
            numbers = pd.to_numeric(text.mask(token_mask), errors="coerce")
            bad = int((text.notna() & ~token_mask & numbers.isna()).sum())
            inf = int(np.isinf(numbers).sum())
            if bad:
                invalid[c] = bad
            if inf:
                infinite[c] = inf
            numbers = numbers.mask(np.isinf(numbers))                 # inf -> NaN
            if numbers.notna().all() and (numbers % 1 == 0).all():
                numbers = numbers.astype("int64")
            df[c] = numbers
        missing_after = {c: int(n) for c, n in df.isna().sum().items() if n}

        duplicates_removed = 0
        if duplicates and duplicates / original_rows <= MAX_DUPLICATE_FRACTION:
            df = df.drop_duplicates().reset_index(drop=True)
            duplicates_removed = duplicates
        elif duplicates:
            print(f"WARNING: {duplicates} duplicates exceed {MAX_DUPLICATE_FRACTION:.0%}; NOT removed.")
        print(f"Cleaned: {original_rows} -> {len(df)} rows | duplicates removed: {duplicates_removed} "
              f"| invalid numeric: {sum(invalid.values())} | infinite->NaN: {sum(infinite.values())} "
              f"| placeholders->NaN: {placeholders}")
        if missing_after:
            print("Missing values remain (rows kept):", missing_after)

        # ---- 6. Save ----
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{split}_cleaned.csv", index=False)
        print("Saved:", out_dir / f"{split}_cleaned.csv")

        report[split] = {
            "input_file": str(path),
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "duplicates_found": duplicates,
            "duplicates_removed": duplicates_removed,
            "invalid_numeric_values": invalid,
            "infinite_values_replaced": infinite,
            "placeholder_values_replaced": placeholders,
            "missing_values_after_cleaning": missing_after,
            "label_column": label_col,
            "categorical_columns": categorical_cols,
            "label_distribution_before": label_before,
            "label_distribution_after": df[label_col].fillna("<missing>").value_counts().to_dict(),
            "note": "Remaining NaNs: impute using TRAIN statistics only. Do not use label/difficulty_level "
                    "as features. Encoding and scaling belong to the preprocessing stage.",
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=4, default=str), encoding="utf-8")
    print("\nReport saved:", report_path)


if __name__ == "__main__":
    clean_data()