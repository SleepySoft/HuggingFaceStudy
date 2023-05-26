import torch
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)

model.save_pretrained("directory_on_my_computer")

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)

print(output)
