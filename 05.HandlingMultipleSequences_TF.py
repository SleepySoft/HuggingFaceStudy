import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# input_ids = tf.constant(ids)
# This line will go error.
# model(input_ids)
# No error information has been found
# Because the input_ids should be a list of ids

input_ids = tf.constant([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# # ------------------------------------- Manually Padding ID -------------------------------------

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# ---- Wrong because mask is not specified, the model will process the padding ----

print(model(tf.constant(sequence1_ids)).logits)
print(model(tf.constant(sequence2_ids)).logits)
print(model(tf.constant(batched_ids)).logits)

# --------------- Specify the mask to make model ignore the padding ---------------

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(tf.constant(batched_ids), attention_mask=tf.constant(attention_mask))
print(outputs.logits)
