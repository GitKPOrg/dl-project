# src/model_BERT.py
"""
Training & evaluation helper for a given model.
This file does not do data prep. It expects tokenized HF Datasets (train_tok, val_tok, test_tok)
either in memory or saved to disk (loadable via load_from_disk).

Use the notebook to set MODEL_NAME, num_labels, and training_args, then call train_and_evaluate(...)
"""
#['lang']
import os
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from src.utils import save_trainer_history_to_csv, save_results_csv, trainer_train_with_timing, evaluate_trainer_with_timing, make_run_dir

def train_and_evaluate(
    train_tok: Optional[Any] = None,
    val_tok: Optional[Any] = None,
    test_tok: Optional[Any] = None,
    train_tok_path: Optional[str] = None,
    val_tok_path: Optional[str] = None,
    test_tok_path: Optional[str] = None,
    MODEL_NAME: str = "distilbert-base-uncased",
    num_labels: int = 3,
    training_args_overrides: Optional[Dict[str, Any]] = None,
    output_root: str = "outputs",
    run_name_suffix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train & evaluate given tokenized datasets with the chosen MODEL_NAME.
    - If train_tok/val_tok/test_tok in memory are provided, these are used.
    - Else, if paths provided, datasets will be loaded via load_from_disk.
    - training_args_overrides: dict with per_device_train_batch_size, eval_batch_size, num_train_epochs, learning_rate, logging_steps etc.
    Returns res_train dict with trainer, eval metrics, and run_outdir.
    """

    # load from disk if needed
    if train_tok is None and train_tok_path is not None:
        train_tok = load_from_disk(train_tok_path)
    if val_tok is None and val_tok_path is not None:
        val_tok = load_from_disk(val_tok_path)
    if test_tok is None and test_tok_path is not None:
        test_tok = load_from_disk(test_tok_path)

    if train_tok is None or val_tok is None or test_tok is None:
        raise ValueError("train_tok, val_tok, test_tok must be provided (either as objects or paths).")

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # prepare training args with sensible defaults; override from user dict
    defaults = {
        "output_dir": "/tmp/hf_temp_output",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
        "logging_steps": 100,
    }
    if training_args_overrides:
        defaults.update(training_args_overrides)

    training_args = TrainingArguments(
        output_dir=defaults["output_dir"],
        eval_strategy=defaults["eval_strategy"],
        save_strategy=defaults["save_strategy"],
        per_device_train_batch_size=defaults["per_device_train_batch_size"],
        per_device_eval_batch_size=defaults["per_device_eval_batch_size"],
        num_train_epochs=defaults["num_train_epochs"],
        learning_rate=defaults["learning_rate"],
        weight_decay=defaults["weight_decay"],
        fp16=defaults["fp16"],
        logging_steps=defaults["logging_steps"],
        push_to_hub=False
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # make run dir with neat name
    hp = {"learning_rate": training_args.learning_rate,
          "per_device_train_batch_size": training_args.per_device_train_batch_size,
          "num_train_epochs": training_args.num_train_epochs}
    run_outdir = make_run_dir(output_root, MODEL_NAME, hp)
    if run_name_suffix:
        run_outdir = run_outdir + f"_{run_name_suffix}"
        os.makedirs(run_outdir, exist_ok=True)
    trainer.args.output_dir = run_outdir

    # train (timed)
    train_res = trainer_train_with_timing(trainer)
    train_seconds = train_res.get("train_seconds", None)

    # save history CSV
    hist_csv = os.path.join(run_outdir, "trainer_history.csv")
    save_trainer_history_to_csv(trainer, hist_csv)

    # evaluate on test (timed)
    eval_res = evaluate_trainer_with_timing(trainer, test_tok)

    # save classification report and confusion matrix
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        pred_out = trainer.predict(test_tok)
        y_true = pred_out.label_ids
        y_pred = pred_out.predictions.argmax(axis=-1)
        import json
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        with open(os.path.join(run_outdir, "classification_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        with open(os.path.join(run_outdir, "confusion_matrix.json"), "w", encoding="utf-8") as fh:
            json.dump({"cm": cm.tolist()}, fh, indent=2)
    except Exception as e:
        print("Warning: could not compute/save classification report:", e)

    # save results CSV (team summary)
    results_csv = os.path.join(output_root, "team_comparison_results.csv")
    save_results_csv({"n_samples": len(test_tok),
                      "pred_time_seconds": eval_res.get("pred_time_seconds"),
                      "pred_time_per_sample": eval_res.get("pred_time_per_sample"),
                      "metrics": eval_res.get("metrics"),
                      "train_seconds": train_seconds},
                     results_csv, MODEL_NAME, extra_info={"run_outdir": run_outdir})

    # save model & tokenizer
    trainer.save_model(run_outdir)
    tokenizer.save_pretrained(run_outdir)

    return {
        "run_outdir": run_outdir,
        "trainer": trainer,
        "eval": eval_res,
        "train_seconds": train_seconds,
        "train_tok": train_tok,
        "val_tok": val_tok,
        "test_tok": test_tok
    }
