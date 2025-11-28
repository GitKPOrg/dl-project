'''
model-level config dictionaries here so all models can load shared settings.
'''
# configs/config.py
from datetime import timedelta
#strategy
# Basic shared config used across models
BASE_CONFIG = {
    "seed": 42,
    "max_length": 256,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "fp16": False,  # set True in runtime if CUDA available
    "output_root": "outputs"
}

# Model-specific skeletons (teammates add other models here)
MODEL_CONFIGS = {
    "bert": {
        "model_name": "distilbert-base-uncased",
        "num_labels": 3,   # 3-class by default
        **BASE_CONFIG
    },
    "transformer_from_scratch": {
        "model_name": "transformer_from_scratch_placeholder",
        **BASE_CONFIG
    },
    "fine_tuned_open_weight_llm": {
        "model_name": "open_weight_llm_placeholder",
        **BASE_CONFIG
    },
    "openai_finetuned_llm": {
        "model_name": "openai_finetuned_placeholder",
        **BASE_CONFIG
    },
    "openai_zeroshot": {
        "model_name": "gpt-zero-shot-placeholder",
        **BASE_CONFIG
    }
}
