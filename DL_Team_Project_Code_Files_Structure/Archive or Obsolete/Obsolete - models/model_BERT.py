#9)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_tok = train_bal.map(tokenize_batch, batched=True, remove_columns=["text", "rating"])
val_tok   = val.map(tokenize_batch, batched=True, remove_columns=["text", "rating"])
test_tok  = test.map(tokenize_batch, batched=True, remove_columns=["text", "rating"])

# 10) Model: classification head with 3 labels (for 3-class). For 5-class, change num_labels to 5.
num_labels = 3  # use label3. For 5-class set to 5 and pass label5 instead.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

