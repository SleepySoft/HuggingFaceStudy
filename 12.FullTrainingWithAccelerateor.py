#
# Setup accelerator:
#       accelerate config
# Launch the distributed training:
#       accelerate launch this_file.py
# If using notebook (like Colab)
#       from accelerate import notebook_launcher
#       notebook_launcher(training_function)
#


# +
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

# +
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# +
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()

        # +
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
