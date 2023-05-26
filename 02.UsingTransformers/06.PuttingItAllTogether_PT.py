from transformers import AutoTokenizer, AutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# ------------------- The tokenizer can handle single sequence -------------------

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs)


# ---------------------------- Also multiple sequence ----------------------------

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
print(model_inputs)


# -------------------------------- And do padding --------------------------------

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")
print(model_inputs)

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")
print(model_inputs)

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
print(model_inputs)


# ------------------------------- Also truncating -------------------------------

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)
print(model_inputs)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs)


# ------------------- Conversion to specific framework tensors -------------------

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print(model_inputs)

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")
print(model_inputs)

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print(model_inputs)


# ----------------------- tokenizer will add special words -----------------------

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
print(tokenizer.decode(model_inputs["input_ids"]))

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
print(tokenizer.decode(ids))


# --------------------------- From tokenizer to model ---------------------------

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)
