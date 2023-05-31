import torch
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, AdamW, get_scheduler


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns).
# Rename the column label to labels (because the model expects the argument to be named labels).
# Set the format of the datasets so they return PyTorch tensors instead of lists.

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


# To quickly check there is no mistake in the data processing, we can inspect a batch like this:

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


optimizer = AdamW(model.parameters(), lr=5e-5)


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# scheduler 是学习率调度器的一个实例。学习率调度器用于在训练过程中动态调整优化器的学习率。
#       不同的学习率调度器使用不同的策略来调整学习率，例如线性衰减、余弦衰减、分段常数等。
# 在这段代码中，我们使用 get_scheduler 函数创建了一个线性学习率调度器。这个调度器会在训练过程中将学习率从初始值线性衰减到 0。
# lr_scheduler.step() 函数用于更新学习率。在每次更新模型参数后，我们都需要调用这个函数来更新优化器的学习率。这样，优化器在下一次更新模型参数时就会使用新的学习率。

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# Q: Adam不是会自动调整梯度下降的幅度吗？为什么我们还需要学习率调度器？
# A: 是的，Adam 优化器会根据梯度的一阶矩和二阶矩动态调整每个参数的学习率。这种自适应学习率调整方法可以加速模型的收敛，并且能够适应不同的数据分布和模型结构。
#    然而，尽管 Adam 优化器能够自适应地调整每个参数的学习率，但它仍然需要一个全局学习率作为基准。
#    全局学习率决定了 Adam 优化器更新每个参数时学习率的上限。因此，我们仍然需要使用学习率调度器来动态调整全局学习率。
#    在实践中，我们通常会在训练过程中逐渐降低全局学习率，以便在模型接近最优解时能够更精细地调整模型参数。这就是为什么我们需要使用学习率调度器的原因。


# ---------------------------------- The training loop ----------------------------------

# 这段代码中的训练方法与前面提到的 Trainer 类的训练方法不同。这段代码中，我们手动定义了一个训练循环，用于在指定的 epoch 数内对模型进行训练。
# 在每个 epoch 中，我们遍历训练数据集中的每个批次，并使用模型计算损失。然后，我们使用反向传播算法计算梯度，并使用优化器更新模型参数。
#
# 而在前面提到的 Trainer 类中，训练过程是自动进行的。我们只需要创建一个 Trainer 对象，并指定训练数据集、验证数据集、数据整理器、分词器和评估函数等参数。
# 然后，我们可以使用 trainer.train() 函数开始训练模型。Trainer 类会自动执行训练循环，并在每个 epoch 结束时根据指定的评估策略对模型进行评估。
#
# 使用 Trainer 类和手动定义训练循环这两种方法在效率上没有太大差别。它们的主要区别在于易用性和灵活性。
# 使用 Trainer 类的优点是它非常易用。我们只需要创建一个 Trainer 对象，并指定训练数据集、验证数据集、数据整理器、分词器和评估函数等参数。
# 然后，我们可以使用 trainer.train() 函数开始训练模型。Trainer 类会自动执行训练循环，并在每个 epoch 结束时根据指定的评估策略对模型进行评估。
#
# 手动定义训练循环的优点是它非常灵活。我们可以根据需要自定义训练循环中的每一个步骤，例如计算损失、更新模型参数、调整学习率等。
# 这样，我们就可以实现一些特殊的训练策略，或者在训练过程中添加一些自定义的操作。
#
# 总之，如果你想快速实现一个简单的训练过程，那么使用 Trainer 类可能是一个更好的选择。如果你需要实现一些特殊的训练策略，或者想要对训练过程进行更多的控制，那么手动定义训练循环可能更合适。


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)


progress_bar = tqdm(range(num_training_steps))

# 将 PyTorch 模型设置为训练模式。在训练模式下，模型的行为可能会与评估模式不同。
# 例如，某些层（如批量归一化层和丢弃层）在训练和评估模式下的行为不同。因此，在训练模型之前，我们需要调用 model.train() 来确保模型处于正确的状态。
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        # optimizer.zero_grad() 函数用于清除优化器中累积的梯度。在每次更新模型参数之前，我们都需要调用这个函数来清除上一次迭代中计算的梯度。
        # 在 PyTorch 中，梯度是累积的，也就是说每次调用 loss.backward() 函数时，计算出的梯度都会累加到模型参数的 grad 属性中。
        # 因此，如果我们不清除梯度，那么下一次迭代时计算出的梯度就会叠加在上一次迭代的梯度之上，导致模型无法正确地更新。
        optimizer.zero_grad()

        progress_bar.update(1)


# --------------------------------- The evaluation loop ---------------------------------

metric = evaluate.load("glue", "mrpc")

# 将 PyTorch 模型设置为评估模式。
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    #
    # with torch.no_grad() 是一个上下文管理器，它用于在 PyTorch 中临时禁用自动梯度计算。
    #       这在评估模型时非常有用，因为我们不需要计算梯度，这样可以节省内存并加快计算速度。
    #
    # 的确，在 PyTorch 中，只有在调用 backward() 函数时才会计算梯度。
    #       但是，在计算梯度之前，PyTorch 会在前向传播过程中跟踪计算图并存储中间结果，以便在反向传播时使用。
    #       当我们使用 with torch.no_grad() 时，PyTorch 不会跟踪计算图并存储中间结果，从而节省内存并加快计算速度。
    #
    # 当我们使用模型进行推理（即预测）时，我们通常也会使用 with torch.no_grad() 来禁用自动梯度计算。
    # 这样可以节省内存并加快计算速度，因为在推理过程中我们不需要计算梯度。
    #
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

result = metric.compute()
print(result)
