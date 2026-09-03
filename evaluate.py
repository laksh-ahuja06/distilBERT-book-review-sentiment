# pip install numpy torch datasets transformers scikit-learn 'accelerate>=1.10'
import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

MODEL_NAME = "Lkkash/distilbert-book-reviews"  # Model from huggingface
DATASET_NAME = "Lkkash/book-reviews-from-amazon-and-goodreads" # Dataset from huggingface

TEST_SIZE = 0.01 # percentage to take for test split
SEED = 42

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Using device: {device}")

# Load dataset from huggingface
print("\nLoading dataset...")
dataset = load_dataset(DATASET_NAME)["train"]
print(f"Total examples: {len(dataset)}")

# Create test split
split = dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=SEED,
)

test_dataset = split["test"]
print(f"Test examples: {len(test_dataset)}")

# Run model
all_predictions = []
all_labels = []

batch_size = 16

for i in tqdm(
    range(0, len(test_dataset), batch_size),
    desc="Running predictions"
):
    batch = test_dataset[i:i + batch_size]
    texts = batch["text"]
    labels = batch["label"]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )
    all_predictions.extend(
        predictions.cpu().numpy()
    )
    all_labels.extend(labels)


# Convert to NumPy arrays
predictions = np.array(all_predictions)
labels = np.array(all_labels)

# Calculate all the 4 metrics (accuracy, precision, recall, f1)
accuracy = accuracy_score(
    labels,
    predictions,
)

precision = precision_score(
    labels,
    predictions,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    labels,
    predictions,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    labels,
    predictions,
    average="weighted",
    zero_division=0,
)


# Print all the main metrics
print("\n")
print("=" * 50)
print("           EVALUATION RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# Classification report
print("\n")
print("=" * 50)
print("           CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "Negative",
            "Positive",
        ],
        zero_division=0,
    )
)

# Confusion matrix
print("=" * 50)
print("           CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(
    labels,
    predictions,
)

print("\n              Predicted")
print("              Neg   Pos")
print(
    f"Actual Neg    {cm[0][0]:4d}  {cm[0][1]:4d}"
)
print(
    f"Actual Pos    {cm[1][0]:4d}  {cm[1][1]:4d}"
)
