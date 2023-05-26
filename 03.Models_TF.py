import tensorflow as tf
from transformers import BertConfig, TFBertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = TFBertModel(config)

print(config)

model.save_pretrained("directory_on_my_computer")

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = tf.constant(encoded_sequences)

output = model(model_inputs)

print(output)
