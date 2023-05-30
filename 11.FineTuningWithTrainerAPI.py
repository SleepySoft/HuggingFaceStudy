import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

raw_datasets = load_dataset("glue", "mrpc")


# ----------------------------------------------------------------------------------------------------------------------

# 注意：这里只做了truncation没有做padding，后者是由data_collator做的。
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Only have to provide the trained model saved directory
training_args = TrainingArguments("test-trainer")

# Q: 为什么training_args也需要指定一个保存目录？它和model.save_pretrained有什么不同？
# A: 当使用Trainer类来训练模型时，它会自动在每个epoch结束时保存模型和训练器状态。这样，在训练过程中意外中断时，你可以从上次保存的状态继续训练，而不需要从头开始。
#    当训练完成后，你可以使用 model.save_pretrained 方法来手动保存模型的权重。但是，它并不会保存训练器状态，例如优化器状态、学习率调度器状态等。
#    至于Trainer生成的文件，你可以选择保留或删除这些文件。如果你不打算再次训练这个模型，那么你可以删除这些文件以释放磁盘空间。
#    但是，如果你打算在以后继续训练这个模型，那么建议保留这些文件，以便在需要时从上次保存的状态继续训练。


# Note: The default data_collator used by the Trainer will be a DataCollatorWithPadding if not specified.

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Q: 批次大小是由什么决定的？
# A: 在上面的代码中，批次大小并没有被明确指定。在这种情况下，Trainer 类会使用默认的批次大小，即每个批次包含8个样本。
#    注意：data_collator 是用来将多个样本整合成一个批次的函数。它不决定批次大小，而是根据给定的批次大小来整合样本。
#    可以在创建 TrainingArguments 对象时，通过 per_device_train_batch_size 和 per_device_eval_batch_size 参数来分别指定训练和评估过程中每个设备的批次大小。
#    例如：
#         training_args = TrainingArguments(
#             "test-trainer",
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16
#         )

# Note: We not specify the evaluation_strategy to either "steps" or "epoch".
#       Default to report the training loss every 500 steps

trainer.train()


# ----------------------------------------------------------------------------------------------------------------------

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# predictions.predictions 是一个二维数组，其中每一行表示一个样本，每一列表示一个类别。数组中的值表示模型对每个样本属于每个类别的概率。
# np.argmax(predictions.predictions, axis=-1) 会沿着最后一个轴（即每一行）计算最大值的索引，也就是模型认为每个样本最有可能属于哪个类别。
# 因此，preds 是一个一维数组，其中每个元素表示一个样本的预测类别。

preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

