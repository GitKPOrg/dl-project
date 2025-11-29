# src/common_data_prep.py
"""


Common data preparation for teammates.
NOTE: NO NEED TO RUN THIS. JUST USE CSV FILE TO LOAD DATA AND RUN MODELS.
THIS IS STILL PROVIDED IN CASE YOU ARE INTERESTED TO JUST CHECK OUT HOW WE PREPPED THE DATA.

Provides prepare_data(...) that does steps 1..13:
 - download raw JSONL from HF
 - save raw CSV
 - preprocess combine title+body, extract rating, labels, year
 - filter years 2010..2025
 - keep English only (langdetect)
 - drop empties, cast label, split 80/10/10 stratified
 - oversample train, optional subset / downsample target
 - truncate by words
 - save preprocessed CSV (data/data_train.csv)
 - tokenize (keep text for val/test)
 - save tokenized datasets to run_outdir (timestamped)
 - return tokenized datasets and paths
"""
# ['lang']
import os
from datetime import datetime
from typing import Optional, Dict, Any

from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Import helpers from utils.py (do not modify utils.py)
from src.utils import (
    preprocess_example_batch_combine,
    oversample_dataset,
    truncate_by_words,
    downsample,
    save_hf_dataset_to_csv,
    set_seed
)


def prepare_data(
    repo_id: str,
    chosen_jsonl: str,
    output_root: str = "outputs",
    data_root: str = "data",
    model_name_tag: str = "prep",
    subset_sample: Optional[int] = None,
    downsample_target: Optional[int] = None,
    max_words: int = 256,
    langdetect_ok: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Download, preprocess, tokenize, and save tokenized datasets.

    Returns dict with:
      - run_outdir
      - preproc_csv, raw_csv
      - train_tok, val_tok, test_tok (in memory)
      - train_tok_path, val_tok_path, test_tok_path (on disk)
      - info dict with sizes and config used
    """

    set_seed(seed)

    # 1) download and load raw JSONL
    local_jsonl = hf_hub_download(
        repo_id=repo_id, filename=chosen_jsonl, repo_type="dataset")
    ds = load_dataset("json", data_files=[local_jsonl], split="train")
    # ds_DF1.to_pandas()
    # print("Below is ds_DF1.head")
    # ds_DF1.head()
    print("Loaded dataset:", ds)
    print("column names in raw data", ds.column_names)
    print("length of raw_data", len(ds))
    print("shown below is the ds.head")
    # ds.head() #dataset object has not attribute called head
    ds.select(range(5))
    print("class counts of raw data", ds.to_pandas()["rating"].value_counts())

    # ensure data dir exists
    os.makedirs(data_root, exist_ok=True)
    raw_csv = os.path.join(data_root, "raw_data.csv")
    save_hf_dataset_to_csv(ds, raw_csv)
# res_prep
    # 3) preprocess (combine title/body, extract rating/labels/year)
    # ds = ds.map(preprocess_example_batch_combine, batched=True,
    #             remove_columns=[c for c in ds.column_names if c not in ("text", "rating", "label3", "label5","year")])
    ds = ds.map(preprocess_example_batch_combine, batched=True,
                remove_columns=[c for c in ds.column_names if c not in ("text", "rating", "label3", "label5")])

    # 4) filter years: keep 2010..2025 or None
    ds = ds.filter(lambda x: (x.get("year") is None)
                   or (2010 <= int(x.get("year")) <= 2025))

    # 5) language filter: try langdetect if allowed
    if langdetect_ok:
        try:
            # from langdetect import detect
            # def detect_en_batch(examples):
            #     langs = []
            #     first_key = next(iter(examples))
            #     batch_size = len(examples[first_key])
            #     for i in range(batch_size):
            #         txt = examples.get("text", [""] * batch_size)[i] or ""
            #         try:
            #             lang = detect(txt[:2000]) if txt.strip() else ""
            #         except:
            #             lang = ""
            #         langs.append(lang)
            #     return {"lang": langs}
            # ds = ds.map(detect_en_batch, batched=True, remove_columns=[])

            # #Show columns names - must have langs column
            # print("Columns (in raw data=before filtering for english):", ds.column_names)

            # #Show unique language values
            # unique_langs = ds.unique("lang")
            # print("List of Unique languages found (in raw data=before filtering for english):", unique_langs)

            # #Count how many rows per language
            # from collections import Counter
            # lang_counts = Counter(ds["lang"])
            # print("count of unique languages found (in raw data=before filtering for english)",lang_counts)

            # #print first few rows
            # ds_head1 = ds.select(range(5))
            # print("First few rows (in raw data=before filtering for english):")
            # print(ds_head1)

            # #keep only english language rows
            # ds = ds.filter(lambda x: x.get("lang", "") == "en")

            # #Show columns names - must have langs column
            # print("Columns (AFTER filtering for english):", ds.column_names)

            # #Show unique language values
            # unique_langs = ds.unique("lang")
            # print("List of Unique languages found (AFTER filtering for english):", unique_langs)

            # #Count how many rows per language
            # from collections import Counter
            # lang_counts = Counter(ds["lang"])
            # print("count of unique languages found (AFTER filtering for english)",lang_counts)

            # #print first few rows after keeping only english rows
            # ds_head2 = ds.select(range(5))
            # print("First few rows (AFTER filtering for english):")
            # print(ds_head2)
            # -------------------------
            # Robust language detection: try langdetect -> langid -> heuristic
            # -------------------------
            try:
                from langdetect import detect as _ld_detect
                _have_langdetect = True
            except Exception:
                _have_langdetect = False

            try:
                import langid as _langid
                _have_langid = True
            except Exception:
                _have_langid = False

            # small English word list for heuristic fallback
            _ENG_COMMON_WORDS = set([
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
                "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
                "she", "or", "an", "will", "my", "one", "all", "would", "there", "their"
            ])

            def is_english_heuristic(text: str, ascii_threshold: float = 0.8, stopword_count: int = 2) -> bool:
                """
                Very simple heuristic: check ascii letter ratio and presence of common english words.
                Returns True if likely English.
                """
                if not text or not isinstance(text, str):
                    return False
                # short-circuit: if text very short, fallback to presence of English stopwords only
                txt = text.strip().lower()
                if len(txt) < 20:
                    matches = sum(
                        1 for w in _ENG_COMMON_WORDS if f" {w} " in f" {txt} ")
                    return matches >= 1

                # ascii ratio
                total = len(txt)
                ascii_count = sum(1 for ch in txt if ord(ch) < 128)
                ascii_ratio = ascii_count / max(1, total)
                # english stopwords presence
                matches = sum(
                    1 for w in _ENG_COMMON_WORDS if f" {w} " in f" {txt} ")
                return (ascii_ratio >= ascii_threshold and matches >= stopword_count) or (matches >= 3)

            def detect_lang_safe(text: str) -> str:
                """
                Try langdetect, then langid, then heuristic.
                Always returns a language code string ('' if unknown).
                """
                if not text or not isinstance(text, str) or text.strip() == "":
                    return ""
                txt = text.strip()
                # 1) langdetect
                if _have_langdetect:
                    try:
                        # detect on up to 2000 chars
                        code = _ld_detect(txt[:2000])
                        if isinstance(code, str) and code:
                            return code
                    except Exception:
                        # continue to langid
                        pass

                # 2) langid
                if _have_langid:
                    try:
                        code, score = _langid.classify(txt[:2000])
                        if isinstance(code, str) and code:
                            return code
                    except Exception:
                        pass

                # 3) heuristic fallback
                try:
                    if is_english_heuristic(txt):
                        return "en"
                except Exception:
                    pass

                return ""

            # batched map function using the safe detector
            def detect_en_batch(examples):
                langs = []
                first_key = next(iter(examples))
                batch_size = len(examples[first_key])
                for i in range(batch_size):
                    txt = examples.get("text", [""] * batch_size)[i] or ""
                    try:
                        langs.append(detect_lang_safe(txt))
                    except Exception:
                        langs.append("")
                return {"lang": langs}

            # Apply mapping and filter English rows
            ds = ds.map(detect_en_batch, batched=True, remove_columns=[])
            ds = ds.filter(lambda x: x.get("lang", "") == "en")
            print("After English filter (robust) rows:", len(ds))

        except Exception:
            # if langdetect missing, keep all rows (safe fallback)
            print("Language detection true but detection failed")
            pass

    # 6) drop empty texts & missing labels
    ds = ds.filter(lambda x: x.get("text") is not None and x.get(
        "text").strip() != "" and x.get("label3") is not None)
    print("length after dropping empty texts and missing labels", len(ds))

    # 7) cast label to ClassLabel and split 80/10/10 stratified
    from datasets import ClassLabel
    num_classes = len(set(ds["label3"]))
    ds = ds.cast_column("label3", ClassLabel(num_classes=num_classes))
    print("length after cast label to label 3", len(ds))
    print("class counts of data before splitting, before oversampling, before downsampling",
          ds.to_pandas()["label3"].value_counts())

    train_and_val = ds.train_test_split(
        test_size=0.10, stratify_by_column="label3", seed=seed)
    train_and_val_d = train_and_val["train"]
    test = train_and_val["test"]
    tmp = train_and_val_d.train_test_split(
        test_size=0.1111, stratify_by_column="label3", seed=seed)
    train = tmp["train"]
    val = tmp["test"]
    print("length of train data after splitting data but before oversampling", len(train))
    print("length of validation data after splitting data but before oversampling", len(val))
    print("length of test data after splitting data but before oversampling", len(test))

    # 8) oversample training
    train_bal = oversample_dataset(train, label_col="label3")
    print("length of train_bal after oversampling", len(train_bal))
    print("class counts of oversampled data",
          train_bal.to_pandas()["label3"].value_counts())

    # 9) optional subset_sample (fast debug)
    if subset_sample is not None and subset_sample < len(train_bal):
        train_bal = train_bal.select(range(subset_sample))

    # 10) optional downsample to target size
    if downsample_target is not None and len(train_bal) > int(downsample_target):
        train_bal = downsample(train_bal, target_size=int(
            downsample_target), col_stratified="label3", seed=seed)
    print("train_bal columns:", train_bal.column_names)
    print("length of train_bal after downsampling", len(train_bal))
    print("class counts of downsampled data",
          train_bal.to_pandas()["label3"].value_counts())

    # 11) truncate by words
    train_bal = truncate_by_words(train_bal, col="text", max_words=max_words)
    val = truncate_by_words(val, col="text", max_words=max_words)
    test = truncate_by_words(test, col="text", max_words=max_words)
    print("truncate by words completed on train, val and test data")

    # 12) save preprocessed CSV
    # preproc_csv = os.path.join(data_root, "data_train.csv")
    # save_hf_dataset_to_csv(train_bal, preproc_csv, columns=["text", "rating", "label3", "label5", "year", "lang"])
    #
    preproc_csv = os.path.join(data_root, "data_train.csv")
    print("pre-processed csv file path for data_train join completed")
    preproc_csv2 = os.path.join(data_root, "data_val.csv")
    print("pre-processed csv file path for data_val join completed")
    preproc_csv3 = os.path.join(data_root, "data_test.csv")
    print("pre-processed csv file path for data_test join completed")

    # Save only columns that actually exist in train_bal to avoid KeyError
    desired_cols = ["text", "rating", "label3", "label5", "lang"]
    available_cols = [c for c in desired_cols if c in train_bal.column_names]
    if len(available_cols) < len(desired_cols):
        missing = [c for c in desired_cols if c not in available_cols]
        print(
            f"Note: missing columns {missing} — will save only available columns: {available_cols}")

    save_hf_dataset_to_csv(train_bal, preproc_csv, columns=available_cols)
    print("pre-processed csv file saved into data folder")

    info = {
        "train_rows": len(train_bal),
        "val_rows": len(val),
        "test_rows": len(test)
        # "run_outdir": run_outdir
    }

    return {
        # "run_outdir": run_outdir,
        "raw_csv": raw_csv,
        "preproc_csv": preproc_csv,
        "info": info
    }
