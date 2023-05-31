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

# trainer.predict(tokenized_datasets["validation"]) 和直接调用模型会得到相同的预测结果，但是 trainer.predict 方法会执行一些额外的操作。
# 当您使用 trainer.predict 方法时，它会自动处理数据批次，并在每个批次上调用模型进行预测。此外，它还会自动将数据移动到正确的设备（例如 GPU）上，
# 并在预测完成后将结果移回 CPU。最后，它还会计算一些额外的信息，例如预测的损失值。因此，使用 trainer.predict 方法可以让您更方便地对数据集进行预测，并获取一些额外的信息。

# predictions 包含两个属性：predictions.predictions 和 predictions.label_ids。
# predictions.predictions 是一个形状为 (num_examples, num_labels) 的数组，其中 num_examples
#   是验证集中样本的数量，um_labels 是类别的数量。每一行为一个样本在各个标签上的预测概率：
#       [[-3.4582543 ,  3.8360713 ],
#        [ 2.2839937 , -2.8021407 ],
#        [ 1.9941015 , -2.269071  ],
#        [-3.4411662 ,  3.61056   ],
#        [ 2.3833728 , -2.8724198 ]]
# predictions.label_ids 是一个形状为 (num_examples,) 的数组，表示验证集中每个样本的真实标签。

# predictions.predictions 是模型输出的 logits。logits 是模型在最后一层的输出，它表示模型对每个类别的预测得分。
# 在这个例子中，由于是二分类问题，所以 predictions.predictions 的每一行都包含两个值，分别表示模型对两个类别的预测得分。
# 通常情况下，我们会使用 softmax 函数将 logits 转换为概率分布。

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# argmax 是一个函数，它返回输入数组中最大值的索引。
# np.argmax(predictions.predictions, axis=-1) 会沿着最后一个轴（即每一行）计算最大值的索引。即概率最高的那个分类。
# 因此，preds 是一个一维数组，其中每个元素表示一个样本的预测类别。

preds = np.argmax(predictions.predictions, axis=-1)
print(preds)

# 加载 GLUE 数据集中 MRPC 任务的评估指标。对于 MRPC 任务，评估指标是 F1 分数和准确率。
#   F1 分数是一种用于衡量分类模型性能的指标，它是精确率和召回率的调和平均值。精确率表示在所有被模型预测为正类的样本中，真实为正类的比例；召回率表示在所有真实为正类的样本中，被模型预测为正类的比例。
#   F1 分数的计算公式为：F1 = 2 * 精确率 * 召回率 / (精确率 + 召回率)
#   F1 分数的取值范围为 0 到 1，值越大表示模型性能越好。当精确率和召回率都很高时，F1 分数也会很高。因此，F1 分数可以用来衡量模型在同时保证精确率和召回率方面的能力。
metric = evaluate.load("glue", "mrpc")

# 使用 compute 方法来计算模型在验证集上的 F1 分数和准确率。它接受两个参数：predictions 是模型的预测结果，references 是验证集的真实标签。
result = metric.compute(predictions=preds, references=predictions.label_ids)
print(result)


# -------------------------------------------- Evaluate metric in training ---------------------------------------------

metric = evaluate.load("glue", "mrpc")


def compute_metrics(eval_preds):
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

