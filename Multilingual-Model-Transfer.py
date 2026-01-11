from datasets import load_dataset
from collections import Counter
import evaluate
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -------------------------
# Model
# -------------------------
model_name = "xlm-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# -------------------------
# Tokenization
# -------------------------
def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True
    )


# -------------------------
# Load datasets
# -------------------------
ds_en = load_dataset("xnli", "en")
ds_hi = load_dataset("xnli", "hi")

ds_en["train"] = ds_en["train"].shuffle(seed=42).select(range(2000))
ds_en["validation"] = ds_en["validation"].shuffle(seed=42).select(range(500))
ds_hi["train"] = ds_hi["train"].shuffle(seed=42).select(range(500))




ds_en = ds_en.map(tokenize, batched=True)
ds_hi = ds_hi.map(tokenize, batched=True)


# -------------------------
# Data collator (THIS FIXES YOUR ERROR)
# -------------------------
data_collator = DataCollatorWithPadding(tokenizer)

# -------------------------
# Training arguments
# -------------------------
args = TrainingArguments(
    output_dir="nli_out",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)



ds_en.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

ds_hi.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)   
# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_en["train"],
    eval_dataset=ds_en["validation"],
    data_collator=data_collator,
    compute_metrics = compute_metrics,
)

print("Starting training NOW...")
trainer.train()

model.save_pretrained("xnli_model", safe_serialization=False)
tokenizer.save_pretrained("xnli_model")
print("XNLI model saved!")


print(Counter(ds_en["validation"]["label"]))
trainer.evaluate(ds_en["validation"])
trainer.evaluate(ds_hi["validation"])


metrics = trainer.evaluate()
print(metrics)

# ---------- EN → EN ----------
metrics_en = trainer.evaluate(ds_en["validation"])
print("EN → EN metrics:", metrics_en)

preds_en = trainer.predict(ds_en["validation"])
y_pred_en = np.argmax(preds_en.predictions, axis=1)
y_true_en = preds_en.label_ids

from sklearn.metrics import confusion_matrix
print("EN → EN confusion matrix:")
print(confusion_matrix(y_true_en, y_pred_en))


# ---------- EN → HI ----------
metrics_hi = trainer.evaluate(ds_hi["validation"])
print("EN → HI metrics:", metrics_hi)

preds_hi = trainer.predict(ds_hi["validation"])
y_pred_hi = np.argmax(preds_hi.predictions, axis=1)
y_true_hi = preds_hi.label_ids

print("EN → HI confusion matrix:")
print(confusion_matrix(y_true_hi, y_pred_hi))



