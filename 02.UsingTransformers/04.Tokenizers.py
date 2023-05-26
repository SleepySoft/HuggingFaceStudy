from transformers import BertTokenizer, AutoTokenizer

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

result = tokenizer("Using a Transformer network is simple")
print(result)

tokenizer.save_pretrained("directory_on_my_computer")

# ---------------------------------------------------------------------------------

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

# ---------------------------------------------------------------------------------

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# ---------------------------------------------------------------------------------

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
