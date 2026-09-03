# Important libraries to install
# pip install transformers scikit-learn datasets pandas numpy torch 'accelerate>=1.10'
# OR use pip install -r requirements.txt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import numpy as np
import torch

FILE_NAME = "books_only_reviews_clean.csv"  ## Dataset (saved locally during finetuning)
MODEL_NAME = "distilbert-base-uncased"      ## Base model to finetune
NUM_LABELS = 2                              ## Total number of different labels (0,1 for book-review sentiment)

# Load model and tokenizer
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

# Load data
df = pd.read_csv(FILE_NAME)
# Keep only the columns we need
df = df[["text", "label"]]

# Rename label -> labels because Hugging Face Trainer expects the target column to be called "labels"
df = df.rename(columns={"label": "labels"})
# Make sure labels are integers
df["labels"] = df["labels"].astype(int)
print(df.head(10))
print(df["labels"].value_counts())


# Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Train / validation split
# Train = The dataset the model learns from to find patterns.
# Validation = The dataset used to tune settings and stop the model from memorizing.
# Test = The final dataset used to unbiasedly grade the finished model.

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=df["labels"]       # Keeps 0/1 proportions similar
)

# Convert to huggingFace datasets
train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

# Tokenize
train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# Tell Hugging Face which columns to return as tensors
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
  
    # For single-label classification: choose the class with the highest logit
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="binary"
        ),
        "precision": precision_score(
            labels,
            predictions,
            average="binary",
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    }

# Device (mps for mac)
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print("Using device:", device)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./book_sentimental",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    logging_steps=100,
    report_to="none",
    fp16=False,
    bf16=False,
    seed=42,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# Train the model
trainer.train()

# Save the model
trainer.save_model("./distilbert-book-reviews")
tokenizer.save_pretrained(
    "./distilbert-book-reviews"
)
