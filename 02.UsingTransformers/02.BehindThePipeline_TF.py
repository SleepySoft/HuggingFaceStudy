# from transformers import pipeline


# classifier = pipeline("sentiment-analysis")
# classifier(
#     [
#         "I've been waiting for a HuggingFace course my whole life.",
#         "I hate this so much!",
#     ]
# )


import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
print('-------------------------------------- %s --------------------------------------' % 'inputs as tokenizer output')
print(inputs)
print()


# ------------------------------------------------------------------------------------------------

model = TFAutoModel.from_pretrained(checkpoint)
outputs = model(inputs)

print('-------------------------------------- %s --------------------------------------' %
      'AutoModel outputs.last_hidden_state.shape')
print(outputs.last_hidden_state.shape)
print()

print('-------------------------------------- %s --------------------------------------' %
      'AutoModel outputs.last_hidden_state')
print(outputs.last_hidden_state)
print()

print('-------------------------------------- %s --------------------------------------' % 'model.config.id2label')
print(model.config.id2label)
print()


# ------------------------------------------------------------------------------------------------

#
# Model Architecture
# Model (retrieve the hidden states)
# ForCausalLM
# ForMaskedLM
# ForMultipleChoice
# ForQuestionAnswering
# ForSequenceClassification
# ForTokenClassification
# and others 🤗
#

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(inputs)

print('-------------------------------------- %s --------------------------------------' %
      'AutoModelForSequenceClassification outputs.logits.shape')
print(outputs.logits.shape)
print()

print('-------------------------------------- %s --------------------------------------' %
      'AutoModelForSequenceClassification outputs.logits')
print(outputs.logits)
print()

print('-------------------------------------- %s --------------------------------------' % 'model.config.id2label')
print(model.config.id2label)
print()


# ------------------------------------------------------------------------------------------------

predictions = tf.math.softmax(outputs.logits, axis=-1)
print('-------------------------------------- %s --------------------------------------' % 'predictions')
print(predictions)
print()
