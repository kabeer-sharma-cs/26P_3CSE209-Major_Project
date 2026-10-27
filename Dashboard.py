import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths (same layout the project scripts already use)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"
REPORTS = ROOT / "outputs" / "reports"

MODEL_PATH = MODELS / "nids_model.keras"
PREPROCESSOR_PATH = MODELS / "preprocessor.joblib"
TRAIN_CLEANED = DATA / "train_cleaned.csv"
X_TEST_PATH = DATA / "X_test.npy"
Y_TEST_PATH = DATA / "y_test.npy"

PIPELINE = [
    ("Data cleaning", "Data_Cleaning.py", [TRAIN_CLEANED]),
    ("Preprocessing", "Data_Preprocessing.py", [PREPROCESSOR_PATH, X_TEST_PATH, Y_TEST_PATH]),
    ("Model training", "Model_Training.py", [MODEL_PATH]),
    ("Evaluation", "model_evaluation.py", [REPORTS / "evaluation_metrics.json"]),
]

NORMAL_HINTS = {"normal", "benign", "legitimate", "0"}

# ---------------------------------------------------------------------------
# Page setup and light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-left: 4px solid #0f8b8d;
        padding: .7rem .9rem; border-radius: 6px;
    }
    .verdict-ok  {border-left: 5px solid #2e9e5b; padding: .6rem 1rem; background: rgba(46,158,91,.10); border-radius: 4px;}
    .verdict-bad {border-left: 5px solid #d64545; padding: .6rem 1rem; background: rgba(214,69,69,.10); border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_cached(path_str, mtime):
    from keras.models import load_model
    return load_model(path_str)


@st.cache_resource
def load_preprocessor_cached(path_str, mtime):
    return joblib.load(path_str)


@st.cache_data
def load_test_arrays(mtime):
    return np.load(X_TEST_PATH), np.load(Y_TEST_PATH).astype(int).squeeze()


@st.cache_data
def load_train_cleaned(mtime):
    return pd.read_csv(TRAIN_CLEANED)


@st.cache_data(show_spinner="Predicting on test set...")
def predict_test(model_mtime, test_mtime):
    model = load_model_cached(str(MODEL_PATH), model_mtime)
    X_test, _ = load_test_arrays(test_mtime)
    prob = model.predict(X_test, verbose=0)
    return prob


@st.cache_data
def clean_rows_and_test_index(csv_mtime, pre_mtime):
    from sklearn.model_selection import train_test_split

    pre = load_preprocessor_cached(str(PREPROCESSOR_PATH), pre_mtime)
    df = load_train_cleaned(csv_mtime)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Label"])
    mapping = {name: i for i, name in enumerate(pre["classes"])}
    y = df["Label"].map(mapping).to_numpy()
    _, test_idx = train_test_split(df.index.to_numpy(), test_size=0.2, random_state=42, stratify=y)
    return df, set(test_idx.tolist())


def mtime(path):
    return path.stat().st_mtime if path.exists() else 0


def read_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def feature_names(pre):
    scaler = pre["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return list(pre["median"].index)


def is_attack(name):
    return str(name).strip().lower() not in NORMAL_HINTS


def pipeline_status():
    return [(label, script, all(p.exists() for p in outputs)) for label, script, outputs in PIPELINE]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ NIDS Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Dataset", "Training", "Evaluation", "Live prediction", "Run pipeline"],
)
st.sidebar.divider()
for label, _, done in pipeline_status():
    st.sidebar.write(("✅ " if done else "⬜ ") + label)
st.sidebar.caption(f"Project folder:\n{ROOT}")

model_ready = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
test_ready = X_TEST_PATH.exists() and Y_TEST_PATH.exists()


def need(message):
    st.warning(message + " Go to **Run pipeline** to create it.")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Network Intrusion Detection")
    st.write("A neural network that classifies network traffic as normal or as a type of attack.")

    metrics = read_json(REPORTS / "evaluation_metrics.json")
    if metrics:
        c = st.columns(4)
        c[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c[1].metric("Macro precision", f"{metrics['precision_macro']:.2%}")
        c[2].metric("Macro recall", f"{metrics['recall_macro']:.2%}")
        c[3].metric("Macro F1", f"{metrics['f1_macro']:.2%}")
        st.caption(f"Measured on {metrics['num_test_samples']:,} held-out test samples.")
    else:
        need("No evaluation results yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Pipeline status")
        status = pd.DataFrame(
            [{"Step": l, "Script": s, "Status": "Done" if d else "Not run"} for l, s, d in pipeline_status()]
        )
        st.dataframe(status, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Model")
        if model_ready:
            pre = load_preprocessor_cached(str(PREPROCESSOR_PATH), mtime(PREPROCESSOR_PATH))
            model = load_model_cached(str(MODEL_PATH), mtime(MODEL_PATH))
            st.write(f"**Classes ({len(pre['classes'])}):** " + ", ".join(map(str, pre["classes"])))
            st.write(f"**Input features:** {model.input_shape[-1]}")
            st.write(f"**Parameters:** {model.count_params():,}")
            st.write("**Architecture:** Dense 128 → 64 → 32 → softmax, with dropout 0.3")
        else:
            need("No trained model found.")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    st.title("Dataset")
    cleaning = read_json(REPORTS / "cleaning_report.json")

    if cleaning:
        st.subheader("Cleaning summary")
        rows = []
        for split, r in cleaning.items():
            rows.append({
                "Split": split,
                "Original rows": r["original_rows"],
                "Cleaned rows": r["cleaned_rows"],
                "Duplicates found": r["duplicates_found"],
                "Duplicates removed": r["duplicates_removed"],
                "Invalid numeric": sum(r["invalid_numeric_values"].values()),
                "Infinite → NaN": sum(r["infinite_values_replaced"].values()),
                "Columns with missing values": len(r["missing_values_after_cleaning"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        need("No cleaning report found.")

    if TRAIN_CLEANED.exists():
        df = load_train_cleaned(mtime(TRAIN_CLEANED))
        label_col = "Label" if "Label" in df.columns else df.columns[-1]

        c1, c2 = st.columns([3, 2])
        counts = df[label_col].value_counts()
        with c1:
            st.subheader("Class distribution")
            st.bar_chart(counts)
        with c2:
            st.subheader("Class share")
            share = (counts / counts.sum()).rename("Share")
            table = pd.DataFrame({"Rows": counts, "Share": share.map("{:.2%}".format)})
            st.dataframe(table, use_container_width=True)

        st.subheader("Explore a feature")
        numeric = df.select_dtypes("number").columns.tolist()
        if numeric:
            fc1, fc2 = st.columns(2)
            feature = fc1.selectbox("Feature", numeric)
            classes = sorted(df[label_col].unique())
            chosen = fc2.multiselect("Classes", classes, default=classes[:3])
            subset = df[df[label_col].isin(chosen)]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            for name, grp in subset.groupby(label_col):
                values = grp[feature].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values):
                    ax.hist(np.log1p(values.clip(lower=0)), bins=40, alpha=0.5, label=str(name))
            ax.set_xlabel(f"log(1 + {feature})")
            ax.set_ylabel("Rows")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Sample rows")
        st.caption("Random rows, because the file may be ordered by class and the first rows would be misleading.")
        if "sample_seed" not in st.session_state:
            st.session_state.sample_seed = 0
        if st.button("Draw new sample"):
            st.session_state.sample_seed += 1
        st.dataframe(
            df.sample(min(100, len(df)), random_state=st.session_state.sample_seed),
            use_container_width=True,
        )

        top_share = counts.iloc[0] / counts.sum()
        if top_share > 0.7:
            st.warning(
                f"'{counts.index[0]}' makes up {top_share:.0%} of the data. Accuracy alone will look "
                "good even if attacks are missed, so check per-class recall on the Evaluation page."
            )
    else:
        need("Cleaned training data not found.")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
elif page == "Training":
    st.title("Training")
    acc, loss = PLOTS / "accuracy.png", PLOTS / "loss.png"
    if acc.exists() and loss.exists():
        c1, c2 = st.columns(2)
        c1.image(str(acc), caption="Accuracy per epoch")
        c2.image(str(loss), caption="Loss per epoch")
        st.caption("A widening gap between training and validation lines suggests overfitting.")
    else:
        need("Training plots not found.")

    if model_ready:
        st.subheader("Model layers")
        model = load_model_cached(str(MODEL_PATH), mtime(MODEL_PATH))
        layer_rows = [
            {"Layer": l.name, "Type": l.__class__.__name__, "Output shape": str(l.output.shape), "Parameters": l.count_params()}
            for l in model.layers
        ]
        st.dataframe(pd.DataFrame(layer_rows), hide_index=True, use_container_width=True)
        st.write("**Settings:** 20 epochs max, batch size 64, Adam, early stopping (patience 3), balanced class weights.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Evaluation on the held-out test set")

    if not (model_ready and test_ready):
        need("Model or test data not found.")
    else:
        pre = load_preprocessor_cached(str(PREPROCESSOR_PATH), mtime(PREPROCESSOR_PATH))
        classes = [str(c) for c in pre["classes"]]
        labels = list(range(len(classes)))
        _, y_test = load_test_arrays(mtime(X_TEST_PATH))
        prob = predict_test(mtime(MODEL_PATH), mtime(X_TEST_PATH))
        y_pred = prob.argmax(axis=1)

        c = st.columns(4)
        c[0].metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        c[1].metric("Macro precision", f"{precision_score(y_test, y_pred, labels=labels, average='macro', zero_division=0):.2%}")
        c[2].metric("Macro recall", f"{recall_score(y_test, y_pred, labels=labels, average='macro', zero_division=0):.2%}")
        c[3].metric("Macro F1", f"{f1_score(y_test, y_pred, labels=labels, average='macro', zero_division=0):.2%}")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Confusion matrix")
            normalize = st.toggle("Show as % of each actual class", value=False)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            if normalize:
                shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
                fmt = ".1%"
            else:
                shown, fmt = cm, "d"
            size = max(6, len(classes) * 0.8)
            fig, ax = plt.subplots(figsize=(size, size * 0.8))
            sns.heatmap(shown, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader("Per-class scores")
            p = precision_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
            r = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
            f = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
            support = np.bincount(y_test, minlength=len(classes))
            per_class = pd.DataFrame({"Precision": p, "Recall": r, "F1": f, "Support": support}, index=classes)
            st.dataframe(
                per_class.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                use_container_width=True,
            )
            st.bar_chart(per_class[["Precision", "Recall", "F1"]])

        # Binary view: normal vs attack, the number a security team cares about
        st.subheader("Normal vs attack")
        actual_attack = np.array([is_attack(classes[i]) for i in y_test])
        pred_attack = np.array([is_attack(classes[i]) for i in y_pred])
        tp = int((actual_attack & pred_attack).sum())
        fn = int((actual_attack & ~pred_attack).sum())
        fp = int((~actual_attack & pred_attack).sum())
        tn = int((~actual_attack & ~pred_attack).sum())
        b = st.columns(4)
        b[0].metric("Attacks caught", f"{tp:,}")
        b[1].metric("Attacks missed", f"{fn:,}")
        b[2].metric("False alarms", f"{fp:,}")
        b[3].metric("Detection rate", f"{tp / max(tp + fn, 1):.2%}")
        st.caption("Any class not named normal / benign is counted as an attack.")

        with st.expander("Saved report from model_evaluation.py"):
            report_csv = REPORTS / "classification_report.csv"
            if report_csv.exists():
                st.dataframe(pd.read_csv(report_csv, index_col=0), use_container_width=True)
            saved_cm = PLOTS / "confusion_matrix.png"
            if saved_cm.exists():
                st.image(str(saved_cm))

# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------
elif page == "Live prediction":
    st.title("Live prediction")

    if not model_ready:
        need("No trained model found.")
    else:
        pre = load_preprocessor_cached(str(PREPROCESSOR_PATH), mtime(PREPROCESSOR_PATH))
        model = load_model_cached(str(MODEL_PATH), mtime(MODEL_PATH))
        classes = [str(c) for c in pre["classes"]]
        cols = feature_names(pre)

        tab1, tab_shuf, tab2 = st.tabs(["Try a test sample", "Shuffled dataset", "Upload a CSV"])

        with tab_shuf:
            st.write(
                "Draws a random, shuffled mix of rows from your cleaned dataset and runs them "
                "through the model, so you see attacks and benign traffic together."
            )
            if not (TRAIN_CLEANED.exists() and "Label" in pd.read_csv(TRAIN_CLEANED, nrows=1).columns):
                need("Cleaned data with a 'Label' column not found.")
            else:
                df_all, test_index = clean_rows_and_test_index(mtime(TRAIN_CLEANED), mtime(PREPROCESSOR_PATH))
                if "shuffle_seed" not in st.session_state:
                    st.session_state.shuffle_seed = 0

                c1, c2, c3 = st.columns(3)
                n_rows = c1.slider("Rows to draw", 10, 500, 50, step=10)
                unseen_only = c2.checkbox("Only rows the model never saw", value=True,
                                          help="The 20% test split. Using training rows would give flattering results.")
                balanced = c3.checkbox("Equal rows per class", value=False,
                                       help="Without this, the mix follows the dataset, which is mostly benign.")

                pool = df_all[df_all.index.isin(test_index)] if unseen_only else df_all
                available = sorted(pool["Label"].unique())
                picked = st.multiselect("Classes to include", available, default=available)
                pool = pool[pool["Label"].isin(picked)]

                if st.button("Shuffle again", type="primary"):
                    st.session_state.shuffle_seed += 1
                seed = st.session_state.shuffle_seed

                if pool.empty:
                    st.info("No rows match these filters.")
                else:
                    if balanced:
                        per_class = max(1, n_rows // pool["Label"].nunique())
                        parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in pool.groupby("Label")]
                        sample = pd.concat(parts)
                    else:
                        sample = pool.sample(min(n_rows, len(pool)), random_state=seed)
                    sample = sample.sample(frac=1, random_state=seed)   # shuffle the final order

                    features = sample[cols].apply(pd.to_numeric, errors="coerce")
                    features = features.replace([np.inf, -np.inf], np.nan).fillna(pre["median"])
                    probs = model.predict(pre["scaler"].transform(features), verbose=0)
                    predicted = [classes[i] for i in probs.argmax(axis=1)]

                    out = pd.DataFrame({
                        "actual": sample["Label"].astype(str).to_numpy(),
                        "predicted": predicted,
                        "confidence": probs.max(axis=1).round(4),
                    }, index=sample.index)
                    out["correct"] = out["actual"] == out["predicted"]

                    m = st.columns(3)
                    m[0].metric("Rows drawn", f"{len(out):,}")
                    m[1].metric("Correct", f"{out['correct'].mean():.1%}")
                    m[2].metric("Wrong", f"{int((~out['correct']).sum()):,}")

                    only_wrong = st.checkbox("Show only mistakes")
                    shown = out[~out["correct"]] if only_wrong else out
                    st.dataframe(
                        shown.style.apply(
                            lambda row: ["background-color: rgba(214,69,69,.18)" if not row["correct"] else "" for _ in row],
                            axis=1,
                        ),
                        use_container_width=True,
                    )
                    st.download_button("Download this sample with predictions",
                                       pd.concat([out, sample.drop(columns="Label")], axis=1).to_csv().encode(),
                                       "shuffled_sample_predictions.csv", "text/csv")

        with tab1:
            if not test_ready:
                need("Test arrays not found.")
            else:
                X_test, y_test = load_test_arrays(mtime(X_TEST_PATH))
                st.write(f"Pick one of the {len(X_test):,} held-out test rows and see what the model says.")
                if "idx" not in st.session_state:
                    st.session_state.idx = 0
                c1, c2 = st.columns([3, 1])
                st.session_state.idx = c1.number_input(
                    "Row number", 0, len(X_test) - 1, st.session_state.idx, step=1
                )
                if c2.button("Random row", use_container_width=True):
                    st.session_state.idx = int(np.random.randint(len(X_test)))
                    st.rerun()

                i = int(st.session_state.idx)
                prob = model.predict(X_test[i:i + 1], verbose=0)[0]
                pred, actual = classes[int(prob.argmax())], classes[int(y_test[i])]

                css = "verdict-ok" if pred == actual else "verdict-bad"
                tag = "Correct" if pred == actual else "Wrong"
                alert = "ATTACK" if is_attack(pred) else "NORMAL"
                st.markdown(
                    f'<div class="{css}"><b>{tag}.</b> Predicted <b>{pred}</b> ({alert}) '
                    f"with {prob.max():.1%} confidence. Actual class: <b>{actual}</b>.</div>",
                    unsafe_allow_html=True,
                )
                st.write("")
                st.bar_chart(pd.Series(prob, index=classes, name="Probability"))
                with st.expander("Scaled feature values"):
                    st.dataframe(pd.DataFrame({"Feature": cols, "Scaled value": X_test[i]}), hide_index=True)

        with tab2:
            st.write(
                f"Upload a CSV with the **{len(cols)} raw feature columns** the model was trained on "
                "(no label column needed). Missing values are filled and values are scaled "
                "automatically with the training statistics."
            )
            with st.expander("Required columns"):
                st.code(", ".join(cols))
            template = pd.DataFrame(columns=cols).to_csv(index=False).encode()
            st.download_button("Download empty template", template, "nids_template.csv", "text/csv")

            upload = st.file_uploader("CSV file", type="csv")
            if upload is not None:
                try:
                    raw = pd.read_csv(upload)
                except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
                    st.error(f"Could not read that file: {err}")
                    st.stop()

                raw.columns = [str(c).strip() for c in raw.columns]
                missing_cols = [c for c in cols if c not in raw.columns]
                if missing_cols:
                    st.error(f"{len(missing_cols)} required column(s) are missing: {missing_cols[:10]}"
                             + (" ..." if len(missing_cols) > 10 else ""))
                    st.stop()

                features = raw[cols].apply(pd.to_numeric, errors="coerce")
                features = features.replace([np.inf, -np.inf], np.nan).fillna(pre["median"])
                scaled = pre["scaler"].transform(features)
                probs = model.predict(scaled, verbose=0)
                result = raw.copy()
                result.insert(0, "prediction", [classes[i] for i in probs.argmax(axis=1)])
                result.insert(1, "confidence", probs.max(axis=1).round(4))
                result.insert(2, "alert", ["ATTACK" if is_attack(p) else "normal" for p in result["prediction"]])

                n_attack = int((result["alert"] == "ATTACK").sum())
                m = st.columns(3)
                m[0].metric("Rows scored", f"{len(result):,}")
                m[1].metric("Flagged as attack", f"{n_attack:,}")
                m[2].metric("Share flagged", f"{n_attack / len(result):.1%}")

                st.bar_chart(result["prediction"].value_counts())
                only_attacks = st.checkbox("Show only flagged rows")
                st.dataframe(result[result["alert"] == "ATTACK"] if only_attacks else result,
                             use_container_width=True)
                st.download_button("Download predictions", result.to_csv(index=False).encode(),
                                   "nids_predictions.csv", "text/csv")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
elif page == "Run pipeline":
    st.title("Run pipeline")
    st.write("Runs the project scripts in order. Each step needs the output of the previous one.")

    steps = pipeline_status()
    chosen = st.multiselect(
        "Steps to run",
        [s for _, s, _ in steps],
        default=[s for _, s, _ in steps],
    )

    if st.button("Run selected steps", type="primary"):
        for _, script, _ in steps:
            if script not in chosen:
                continue
            with st.status(f"Running {script}...", expanded=False) as status:
                result = subprocess.run(
                    [sys.executable, str(ROOT / script)],
                    cwd=ROOT, capture_output=True, text=True, check=False,
                )
                st.code((result.stdout + result.stderr).strip() or "(no output)", language="text")
                if result.returncode != 0:
                    status.update(label=f"{script} failed", state="error")
                    st.error(f"{script} failed. Fix the error above, then run again.")
                    break
                status.update(label=f"{script} finished", state="complete")
        else:
            st.success("All selected steps finished.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.info("Open the other pages to see the new results.")