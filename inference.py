# Dependencies:- pip install torch transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Lkkash/distilbert-book-reviews"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

### Input text ###
text = """
I don't know why i did not understand the book properly, like i don't get what the book was supposed to convey.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.inference_mode():
    outputs = model(**inputs)

logits = outputs.logits

probabilities = torch.softmax(logits, dim=1)

print("\nOutput Probabilities:")
print(f"Bad:  {probabilities[0][0].item() * 100:.2f}%")
print(f"Good: {probabilities[0][1].item() * 100:.2f}%")

prediction = torch.argmax(logits, dim=1).item()

## Convert ids to labels
id_to_label = {
    0: "Bad",
    1: "Good"
}

print("\nPrediction:")
print(id_to_label[prediction])
