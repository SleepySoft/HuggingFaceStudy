import torch
from datasets import load_dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(batch)

# ----------------------------------------------------------------------------------------------------------------------

# This is new

# If "lables" field is added, the loss will be calculated. Otherwise, is None.
batch["labels"] = torch.tensor([1, 1])
print(batch)


loss = model(**batch).loss
print(loss)

param = model.parameters()
optimizer = AdamW(param)

print('-------------- param.grad before backward --------------')
# 注意：model.parameters() 返回的是一个生成器，而不是一个列表或张量
for param in model.parameters():
    print(param.grad)       # 全是None


# 计算损失函数关于模型参数的梯度
loss.backward()

print('-------------- param.grad after backward --------------')
for param in model.parameters():
    print(param.grad)       # 有确切的值

# 根据计算出的梯度更新模型参数。在这个例子中，使用的是 AdamW 优化器。
optimizer.step()

# Q: 这两个函数之间没有参数传递，optimizer.step()怎么知道loss.backward()计算的梯度
# A: 当你调用 loss.backward() 时，PyTorch 会自动计算损失函数关于模型参数的梯度，并将这些梯度存储在模型参数的 .grad 属性中。
#    例如，如果你有一个名为 param 的模型参数，那么在调用 loss.backward() 后，你可以通过 param.grad 来访问它的梯度。
#    当你调用 optimizer.step() 时，优化器会遍历所有已注册的模型参数，并使用它们的 .grad 属性来更新它们的值。
#    在这个例子中，当你创建 AdamW 优化器时，你将模型的参数传递给了它（optimizer = AdamW(model.parameters())），因此优化器知道要更新哪些参数。
#    所以，尽管 loss.backward() 和 optimizer.step() 之间没有直接的参数传递，但它们仍然可以通过模型参数的 .grad 属性来协同工作。

# Q: 每种模型各不相同，optimizer怎么能自动更新所有类型的模型呢？
# A: 不同的模型可能有不同的结构和损失函数。优化器能够自动更新所有类型的模型，是因为它只关心模型参数的梯度，而不关心模型的具体结构或损失函数。
#    无论模型的结构如何，它都有一些可训练的参数，这些参数可以通过调用 model.parameters() 来获取。当你创建优化器时，
#    你将这些参数传递给了它（optimizer = AdamW(model.parameters())），因此优化器知道要更新哪些参数。
#    当你调用 loss.backward() 时，PyTorch 会自动计算损失函数关于模型参数的梯度，并将这些梯度存储在模型参数的 .grad 属性中。
#    优化器不需要知道模型的具体结构或损失函数，它只需要使用这些梯度来更新模型参数。
#    所以，无论模型的结构或损失函数如何，优化器都可以自动更新所有类型的模型，因为它只关心模型参数的梯度。


# ----------------------------------------------------------------------------------------------------------------------

# GLUE 是 General Language Understanding Evaluation 的缩写，旨在推动“研究开发通用且鲁棒的自然语言理解系统”
# “mrpc” 指的是 GLUE 数据集中的一个子任务，即 Microsoft Research Paraphrase Corpus (MRPC)。它是一个用于评估模型在判断两个句子是否具有相同意思方面的能力的任务。
# 所以，当你调用 load_dataset("glue", "mrpc") 时，你实际上是在加载 GLUE 数据集中的 MRPC 任务。

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

# Get dataset's column name and type
print(raw_train_dataset.features)

# Got 2 seperator sequences. Not the expected inputs
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# Got sequence pair inputs, as dictionary and values that are lists of lists
# All data are padding or truncated to the same length
# All data in RAM and not using the Apache Arrow files mechanism
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
# 注意：此时sentence1和sentence2对应位置的句子已经连接在一起作为一个sequence：
#      [[101, sentence1 tokenized, 102, sentence2 tokenized, 102, padding (0)]]
print(tokenized_dataset)


# Use Dataset.map() to keep the data as a dataset
# It returns a new dictionary with the keys input_ids, attention_mask, and token_type_ids
# Note that we have not specified the padding so the data will not be padding to the same length.
def tokenize_function(example):
    result = tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    return result


# Using batched=True so the function is applied to multiple elements of our dataset at once.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)


# ----------------------------------------------------------------------------------------------------------------------

# Using collate function for Dynamic padding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# To check the data have different length
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]


batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

# Collator 是 PyTorch 中用于将多个样本组合成一个批量数据的函数。它通常用于在数据加载器（DataLoader）中对一批样本进行预处理，以便将它们输入到模型中。
# 例如，你可以将 DataCollatorWithPadding 对象传递给 PyTorch 的 DataLoader 类，以便在训练模型时自动对每个批次的数据进行填充。下面是一个简单的例子：
#
# from torch.utils.data import DataLoader
#
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, collate_fn=data_collator)
#
# for batch in dataloader:
#     # batch is a dictionary containing padded sequences
#     # You can input it directly into your model
#     outputs = model(**batch)
