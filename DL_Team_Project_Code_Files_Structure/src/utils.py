'''
This collects the preprocessing, dataset splitting, oversample and metrics utilities from 
your notebook, plus helper functions for output naming and saving. I reused your notebook functions 
and made small fixes for indentation and consistent names so they are importable.
'''
#res_prep
#["lang"]
# src/utils.py
import os
import csv
import time
import random
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import torch
import torch.nn as nn

# ----------------------
# small reproducibility helper
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------
# simple textual preprocess mapping for datasets.map(batched=True)
# keeps consistent output columns: text, rating, label3, label5, year, lang (lang may be None)
# ----------------------
def preprocess_example_batch_combine(examples):
    texts = []
    ratings = []
    label3s = []
    label5s = []
    years = []
    # get batch size
    first_key = next(iter(examples))
    batch_size = len(examples[first_key])
    for i in range(batch_size):
        body = None
        title = None
        # possible body fields
        for f in ("text", "review_body", "review_text", "review"):
            if f in examples and examples.get(f)[i]:
                cand = examples.get(f)[i]
                if cand is not None:
                    body = str(cand).replace("\n", " ").strip()
                    break
        # possible title fields
        for t in ("title", "review_title"):
            if t in examples and examples.get(t)[i]:
                cand = examples.get(t)[i]
                if cand is not None:
                    title = str(cand).replace("\n", " ").strip()
                    break
        if title and body:
            combined = f"{title} — {body}"
        elif body:
            combined = body
        elif title:
            combined = title
        else:
            combined = ""
        combined = " ".join(combined.split()).strip()

        # rating extraction
        rating = None
        for rfield in ("rating", "star_rating", "stars"):
            if rfield in examples and examples.get(rfield)[i] is not None:
                try:
                    rv = examples.get(rfield)[i]
                    rating = int(float(rv))
                except:
                    rating = None
                break

        if rating is None:
            label3 = None
            label5 = None
        else:
            if rating <= 2:
                label3 = 0
            elif rating == 3:
                label3 = 1
            else:
                label3 = 2
            label5 = rating - 1

        # try to read year if present
        year = None
        for yfield in ("review_date", "date", "year"):
            if yfield in examples and examples.get(yfield)[i]:
                try:
                    s = str(examples.get(yfield)[i])
                    # try find 4 digit year
                    import re
                    m = re.search(r"(19|20)\\d{2}", s)
                    if m:
                        year = int(m.group(0))
                except:
                    year = None
                break

        texts.append(combined)
        ratings.append(rating)
        label3s.append(label3)
        label5s.append(label5)
        years.append(year)
    return {"text": texts, "rating": ratings, "label3": label3s, "label5": label5s, "year": years}

# ----------------------
# oversample dataset (return HF Dataset)
# ----------------------
def oversample_dataset(dset: Dataset, label_col: str = "label3") -> Dataset:
    df = dset.to_pandas()
    ros = RandomOverSampler(random_state=42)
    X = df.index.values.reshape(-1, 1)
    y = df[label_col].values
    X_res, y_res = ros.fit_resample(X, y)
    res_idx = X_res.flatten()
    df_res = df.iloc[res_idx].reset_index(drop=True)
    return Dataset.from_pandas(df_res)

# ----------------------
# downsample/back to target size stratified by label column
# ----------------------
from datasets import ClassLabel
def downsample(ds: Dataset, target_size: int, col_stratified: str, seed: int = 42) -> Dataset:
    """
    Casts label column to ClassLabel and returns a stratified downsampled dataset of size target_size.
    """
    num_classes = len(set(ds[col_stratified]))
    label_class = ClassLabel(num_classes=num_classes)
    ds = ds.cast_column(col_stratified, label_class)
    split = ds.train_test_split(train_size=target_size, stratify_by_column=col_stratified, seed=seed, shuffle=True)
    return split["train"]

# ----------------------
# simple truncate by words
# ----------------------
def truncate_by_words(ds: Dataset, col: str = "text", max_words: int = 256) -> Dataset:
    def _truncate(example):
        txt = example.get(col, "") or ""
        words = txt.split()
        example[col] = " ".join(words[:max_words])
        return example
    return ds.map(_truncate)

# ----------------------
# save raw dataframe csv
# ----------------------
def save_hf_dataset_to_csv(ds: Dataset, out_path: str, columns=None):
    df = ds.to_pandas()
    if columns:
        df = df[columns]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV: {out_path}")
    return out_path

# ----------------------
# compute classification metrics (wrapper)
# ----------------------
def compute_classification_metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "classification report": report, "confusion_matrix": cm.tolist()}

# ----------------------
# save run results row to CSV (append)
# ----------------------
def save_results_csv(results: Dict[str, Any], csv_path: str, model_name: str, extra_info: Dict[str, Any] = None):
    row = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "n_samples": results.get("n_samples"),
        "train_seconds": results.get("train_seconds"),
        "pred_time_s": results.get("pred_time_seconds", None),
        "pred_time_s_per_sample": results.get("pred_time_per_sample", None),
        "accuracy": results["metrics"]["accuracy"],
        "precision": results["metrics"]["precision"],
        "recall": results["metrics"]["recall"],
        "f1": results["metrics"]["f1"],
    }
    if extra_info:
        for k, v in extra_info.items():
            row[k] = v
    # store small serialized fields
    row["support_per_class"] = str(results["metrics"].get("support_per_class"))
    row["f1_per_class"] = str(results["metrics"].get("f1_per_class", results["metrics"].get("f1_per_class")))

    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended results to {csv_path}")

# ----------------------
# helper: parse trainer.state.log_history into epoch-level CSV
# ----------------------
def save_trainer_history_to_csv(trainer, out_csv_path: str):
    # trainer.state.log_history is a list of dicts with keys like epoch, loss, eval_loss
    logs = getattr(trainer.state, "log_history", None)
    if not logs:
        print("No trainer log_history found.")
        return None
    rows = []
    # We'll build per-epoch last seen train loss and val loss
    by_epoch = {}
    for entry in logs:
        epoch = entry.get("epoch", None)
        if epoch is None:
            continue
        e = float(epoch)
        if e not in by_epoch:
            by_epoch[e] = {"epoch": e, "train_loss": None, "eval_loss": None, "learning_rate": None}
        if "loss" in entry:
            by_epoch[e]["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            by_epoch[e]["eval_loss"] = float(entry["eval_loss"])
        if "learning_rate" in entry:
            by_epoch[e]["learning_rate"] = float(entry["learning_rate"])
    # sort
    for k in sorted(by_epoch.keys()):
        rows.append(by_epoch[k])
    # save csv
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv_path, index=False)
    print(f"Saved trainer history CSV to {out_csv_path}")
    return out_csv_path

# ----------------------
# focal loss for evaluation convenience
# ----------------------
def focal_loss_from_logits(logits, targets, gamma=2.0, weight=None):
    """
    logits: np.array or torch.tensor raw logits (N, C)
    targets: np.array or torch.tensor labels (N,)
    returns average focal loss (float)
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long)
    ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    logpt = -ce(logits, targets)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return float(loss.mean().item())

# ----------------------------
# Timing helpers
# ----------------------------
import time
from typing import Callable, Dict, Any
from datetime import datetime

def time_function(fn: Callable, *args, **kwargs):
    """
    Run `fn(*args, **kwargs)` and return (result, seconds_elapsed).
    Simple wrapper for quick timing.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def trainer_train_with_timing(trainer, *train_args, **train_kwargs):
    """
    Run trainer.train(...) and return a dict with the raw train output and seconds elapsed.
    Use like: out = trainer_train_with_timing(trainer); out['train_seconds']
    """
    t0 = time.perf_counter()
    train_output = trainer.train(*train_args, **train_kwargs)
    t1 = time.perf_counter()
    sec = t1 - t0
    return {"train_output": train_output, "train_seconds": sec}


# ----------------------------
# Output directory / naming helper
# ----------------------------
import os

def make_run_dir(output_root: str, model_name: str, hyperparams: Dict[str, Any]) -> str:
    """
    Build a run outputs path like:
    outputs/<model_name>_lr{learning_rate}_bs{batch}_ep{epochs}_YYYYmmdd_HHMMSS/
    - safe: replaces '/' in model_name with '_'
    - hyperparams: dict expected to possibly contain keys 'learning_rate', 'lr', 'per_device_train_batch_size',
      'train_batch_size', 'num_train_epochs', 'epochs'. Missing keys become 'NA'.
    """
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr = hyperparams.get("learning_rate", hyperparams.get("lr", "NA"))
    bs = hyperparams.get("per_device_train_batch_size", hyperparams.get("train_batch_size", "NA"))
    ep = hyperparams.get("num_train_epochs", hyperparams.get("epochs", "NA"))
    safe_model = str(model_name).replace("/", "_")
    outdir = os.path.join(output_root, f"{safe_model}_lr{lr}_bs{bs}_ep{ep}_{t}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ----------------------------
# Evaluate trainer with timing and detailed metrics
# ----------------------------
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_trainer_with_timing(trainer, dataset, label_col: str = "labels", batch_size: int = None):
    """
    Run trainer.predict on `dataset`, time it, and return detailed metrics.
    Returns a dict with keys:
      - pred_time_seconds, pred_time_per_sample, n_samples
      - metrics: {accuracy, precision, recall, f1, precision_per_class, recall_per_class, f1_per_class, support_per_class, confusion_matrix}
      - y_true, y_pred, logits
    Notes:
      - trainer must be a HuggingFace Trainer
      - dataset should be tokenized and include label ids
    """
    # remember original batch size, set new if requested
    original_bs = getattr(trainer.args, "per_device_eval_batch_size", None)
    if batch_size is not None:
        trainer.args.per_device_eval_batch_size = batch_size

    t0 = time.perf_counter()
    preds_output = trainer.predict(test_dataset=dataset)
    t1 = time.perf_counter()
    total_seconds = t1 - t0

    logits = preds_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits is None:
        raise ValueError("No predictions returned by trainer.predict().")

    # handle multiclass (2D logits) and binary (1D logits) cases
    if hasattr(logits, "ndim") and logits.ndim == 2:
        y_pred = np.argmax(logits, axis=-1)
    else:
        y_pred = (logits.ravel() > 0.5).astype(int)

    labels = preds_output.label_ids
    if labels is None:
        raise ValueError("No label_ids returned by trainer.predict. Ensure dataset contains labels named 'labels' or returned by original dataset.")

    labels = np.asarray(labels)
    n = len(labels)
    sec_per_sample = total_seconds / n if n > 0 else float("inf")

    # compute metrics
    acc = float(accuracy_score(labels, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(labels, y_pred, average="weighted", zero_division=0)
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(labels, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(labels, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "precision_per_class": prec_per.tolist() if hasattr(prec_per, "tolist") else list(prec_per),
        "recall_per_class": rec_per.tolist() if hasattr(rec_per, "tolist") else list(rec_per),
        "f1_per_class": f1_per.tolist() if hasattr(f1_per, "tolist") else list(f1_per),
        "support_per_class": sup_per.tolist() if hasattr(sup_per, "tolist") else list(sup_per),
        "confusion_matrix": cm.tolist()
    }

    # restore batch_size if modified
    if batch_size is not None and original_bs is not None:
        trainer.args.per_device_eval_batch_size = original_bs

    return {
        "pred_time_seconds": total_seconds,
        "pred_time_per_sample": sec_per_sample,
        "n_samples": int(n),
        "metrics": metrics,
        "y_true": labels,
        "y_pred": y_pred,
        "logits": logits,
    }


"""
Small plotting helpers for loss vs epoch.
"""

from typing import Optional
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def parse_trainer_log_history(log_history):
    """
    Convert trainer.state.log_history (list of dicts) into a DataFrame with last
    training loss and last eval_loss per epoch.
    """
    by_epoch = {}
    for entry in log_history:
        if "epoch" not in entry:
            continue
        ep = float(entry["epoch"])
        if ep not in by_epoch:
            by_epoch[ep] = {"epoch": ep, "train_loss": None, "eval_loss": None}
        if "loss" in entry:
            # store last seen loss for this epoch (we overwrite so last value remains)
            by_epoch[ep]["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            by_epoch[ep]["eval_loss"] = float(entry["eval_loss"])
    if not by_epoch:
        return pd.DataFrame()
    rows = [by_epoch[e] for e in sorted(by_epoch.keys())]
    return pd.DataFrame(rows)


def plot_loss_vs_epochs(trainer=None, hist_csv: Optional[str]=None, run_outdir: Optional[str]=None,
                        save_png: Optional[str]=None, save_csv: Optional[str]=None, show_plot: bool=True):
    """
    Plot training & validation loss per epoch.
    - trainer: HF Trainer object (will use trainer.state.log_history if hist_csv missing)
    - hist_csv: path to a trainer_history CSV (preferred)
    - run_outdir: if provided and save paths not given, images/CSV are saved under run_outdir
    - save_png: explicit path to write PNG
    - save_csv: explicit path to write epoch CSV
    Returns: pandas.DataFrame with columns ['epoch','train_loss','eval_loss'] (may be empty)
    Notes:
      - If fewer than 2 epoch points exist, the function will still plot points but will warn that
        a curve cannot be drawn with <2 points. To get a curve, train for >=2 epochs or log more points.
    """
    df = None

    # 1) try CSV first
    if hist_csv and os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            # ensure columns exist
            if "epoch" not in df.columns:
                df = None
        except Exception:
            df = None

    # 2) fallback: parse trainer.state.log_history
    if df is None and trainer is not None:
        log_history = getattr(getattr(trainer, "state", None), "log_history", None)
        if log_history:
            df = parse_trainer_log_history(log_history)

    # 3) fallback: try to parse trainer_state*.json under run_outdir
    if df is None and run_outdir is not None:
        for p in ("trainer_state.json",):
            cand = os.path.join(run_outdir, p)
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    if isinstance(obj, dict) and "log_history" in obj:
                        df = parse_trainer_log_history(obj["log_history"])
                        break
                except Exception:
                    continue
        # also try any json files in run_outdir
        if df is None:
            for root, _, files in os.walk(run_outdir):
                for f in files:
                    if f.lower().endswith(".json"):
                        cand = os.path.join(root, f)
                        try:
                            with open(cand, "r", encoding="utf-8") as fh:
                                obj = json.load(fh)
                            if isinstance(obj, dict) and "log_history" in obj:
                                df = parse_trainer_log_history(obj["log_history"])
                                break
                        except Exception:
                            continue
                if df is not None:
                    break

    if df is None or df.empty:
        print("No epoch-level losses found (trainer.history or CSV).")
        return pd.DataFrame()

    # Normalize columns
    df = df.sort_values("epoch").reset_index(drop=True)
    if "train_loss" not in df.columns:
        df["train_loss"] = [None] * len(df)
    if "eval_loss" not in df.columns:
        df["eval_loss"] = [None] * len(df)

    # Save epoch CSV if requested
    if save_csv is None and run_outdir is not None:
        save_csv = os.path.join(run_outdir, "loss_per_epoch.csv")
    if save_csv:
        df.to_csv(save_csv, index=False)
        print("Saved epoch CSV to:", save_csv)

    # Prepare PNG path
    if save_png is None and run_outdir is not None:
        save_png = os.path.join(run_outdir, "loss_curve.png")

    # Plot (connect points with lines if at least 2 points)
    epochs = df["epoch"].astype(float).values
    train_loss = df["train_loss"].astype(float).values
    eval_loss = df["eval_loss"].astype(float).values

    plt.figure(figsize=(8,5))
    # check how many non-null train points
    valid_train_idx = ~pd.isna(train_loss)
    valid_eval_idx = ~pd.isna(eval_loss)
    n_train_pts = valid_train_idx.sum()
    n_eval_pts = valid_eval_idx.sum()

    # Plot train loss
    if n_train_pts > 0:
        # if >=2 points, plot with a line; else plot single point
        if n_train_pts >= 2:
            plt.plot(epochs[valid_train_idx], train_loss[valid_train_idx], marker='o', label="train_loss")
        else:
            plt.plot(epochs[valid_train_idx], train_loss[valid_train_idx], marker='o', linestyle='None', label="train_loss (single point)")
    # Plot eval loss
    if n_eval_pts > 0:
        if n_eval_pts >= 2:
            plt.plot(epochs[valid_eval_idx], eval_loss[valid_eval_idx], marker='o', label="eval_loss")
        else:
            plt.plot(epochs[valid_eval_idx], eval_loss[valid_eval_idx], marker='o', linestyle='None', label="eval_loss (single point)")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training & Validation Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_png:
        try:
            plt.savefig(save_png, bbox_inches='tight')
            print("Saved loss plot to:", save_png)
        except Exception as e:
            print("Could not save PNG:", e)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return df
