import pandas as pd
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import transformers
import os
import argparse

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
parser.add_argument('--val', '-v', type=int, help='Path to the output file')
parser.add_argument('--ngpus', '-g', type=int, default=1, help='Path to the output file')
parser.add_argument('--original', '-o', type=int, required=True)
args = parser.parse_args()

transformers.set_seed(42)
torch.backends.cudnn.deterministic = True
suffix = ''

train_gold = pd.read_csv(f"./low_res/100/classification/sst2_train.tsv", sep='\t',header=0)

# unique_labels = list(set(train_gold['label']))

print("Training on gold+augs")
train = pd.read_csv(f"./low_res/100/classification/sst2_train_processed_filtered_generated_augs.source", sep='\t',header=0)

train = train.dropna(subset=['text'])
print(train)
train["text"] = train["text"].apply(lambda x: x.replace('"""',''))

if args.val == 0:
    print("Using test as val")
    dev = pd.read_csv(f"./low_res/100/classification/sst2_dev.tsv", sep='\t',header=0)
else:
    print("Using val")
    dev = pd.read_csv(f"./low_res/100/classification/sst2_dev.tsv", sep='\t',header=0)

test = pd.read_csv(f"./low_res/100/classification/sst2_test.tsv", sep='\t',header=0)

unique_labels = list(set(train['label']))
unique_labels.sort()
# print(unique_labels)
print("Size of train is:",train.shape)
num_labels = len(set(train.iloc[:,1]))
len_train = len(train)
train["text"] = train["text"].apply(lambda x: str(x))

if args.legal == 1:
    model_name = "lexlms/legal-longformer-large"
    learning_rate=1e-5
else:
    model_name = "bert-base-uncased"
    learning_rate=5e-5

print(model_name)

model_ct = AutoModelForSequenceClassification.from_pretrained("./low_res/100/classification/sst2_100", num_labels=num_labels).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

train_ct_df = train.iloc[len(train_gold):]

print(f"Augs len : {len(train_ct_df)}")

train_ct = datasets.Dataset.from_dict({"train":train_ct_df})
inputs = tokenizer(train_ct_df["text"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Define a custom collate function to return a dictionary
def custom_collate(batch):
    return {
        "input_ids": torch.stack([item[0] for item in batch]),
        "attention_mask": torch.stack([item[1] for item in batch]),
    }

dataset = list(zip(inputs["input_ids"], inputs["attention_mask"]))
data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate)

model_ct.eval()

# inputs = {key: value.to("cuda") for key, value in inputs.items()}
# outputs = model_ct(**inputs).logits

predictions = []

print("Starting prediction")

with torch.no_grad():
    for batch in dataloader:
        # input_ids, attention_mask = batch
        # print(input_ids)
        inputs = {"input_ids": batch["input_ids"].to("cuda"), "attention_mask": batch["attention_mask"].to("cuda")}
        outputs = model_ct(**inputs)
        # print(outputs)
        pt_predictions = torch.argmax(outputs[0], dim=1)
        predictions = predictions + list(pt_predictions.detach().cpu().numpy())

# print(predictions)

ct_labels = []
miss = 0
for pred, act, idx in zip(predictions, train_ct_df["label"].tolist(), range(len(predictions))):
    # print(pred)
    pred = unique_labels[pred]
    if pred == act:
        ct_labels.append(idx)
    else:
        miss = miss + 1

print(f"Missing preds : {miss}")

# asdas
# exit()
train_ct_df = train_ct_df.iloc[ct_labels]

train = pd.concat([train_gold, train_ct_df], ignore_index=True)
print(f"Train after consistency : {train}")

train["label"] = train["label"].apply(lambda x: unique_labels.index(x))
dev["label"] = dev["label"].apply(lambda x: unique_labels.index(x))
test["label"] = test["label"].apply(lambda x: unique_labels.index(x))


train = datasets.Dataset.from_dict(train)
dev = datasets.Dataset.from_dict(dev)
test = datasets.Dataset.from_dict(test)
print()
# print(train)
# train["label"] = train["label"].astype(int)

dataset = datasets.DatasetDict({"train":train, "validation":dev, "test":test})
print(dataset)
# print(dataset['train']['label'])
# print(dataset['validation']['label'])
# print(dataset['test']['label'])
# asdasd

tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=False)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

metric = evaluate.load("f1")

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='micro')

num_gpus = args.ngpus
per_device_train_batch_size=int(16/num_gpus)
eval_steps = int(len(train)/(per_device_train_batch_size * num_gpus)) * 5
print(f"Eval steps: {eval_steps}")
training_args = TrainingArguments(
    report_to="none",
    output_dir=f"./test_ckpt/{args.dataset}_{args.split}{suffix}/",
    save_strategy='steps',
    save_steps = eval_steps,
    save_total_limit = 2,
    do_predict = True,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    logging_dir='./logs',
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    metric_for_best_model="eval_f1",
    load_best_model_at_end=True,
    num_train_epochs=20,
    fp16=False
)

print(training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator = data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

preds = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(preds[0], axis=-1)
print(preds, len(preds))

print(metric.compute(predictions=preds, references=test["label"], average='micro'))
print(os.linesep)
print(metric.compute(predictions=preds, references=test["label"], average='macro'))