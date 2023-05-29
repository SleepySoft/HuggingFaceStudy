import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
print(loss)

loss.backward()
optimizer.step()


from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

# Get dataset's column name and type
print(raw_train_dataset.features)


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Got 2 seperator sequences. Not the expected inputs
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# Got sequence pair inputs
# dictionary and values that are lists of lists
# All data in RAM and not using the Apache Arrow files mechanism
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
print(tokenized_dataset)


# Use Dataset.map() to keep the data as a dataset
# It returns a new dictionary with the keys input_ids, attention_mask, and token_type_ids
def tokenize_function(example):
    result = tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    return result


# Using batched=True so the function is applied to multiple elements of our dataset at once.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Using collate function for Dynamic padding

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# To check the data have different lenght
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]


batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
