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
parser.add_argument('--ngpus', '-g', type=int, default=4, help='Path to the output file')
parser.add_argument('--legal', '-l', type=int, required=True)
args = parser.parse_args()

transformers.set_seed(42)
torch.backends.cudnn.deterministic = True

def rename_columns(df):
    # print(df.columns)
    if '0' in df.columns:
        df = df.rename(columns={'0': 'text', '1': 'label'})
    return df

def train_model():

    # gold = rename_columns(pd.read_csv(gold_path, sep='\t', header=0))
    train = rename_columns(pd.read_csv('/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/200/classification/yahoo_train.tsv', sep='\t'))

    print(f"Len of train : {len(train)}")

    # if baseline!="gold" and len(train) < (6*len(gold)-50):
    #     train = pd.concat([gold, train], ignore_index=True, sort=False)
    #     print(f"Len of train after adding gold : {len(train)}")


    dev = rename_columns(pd.read_csv('/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/200/classification/yahoo_dev.tsv', sep='\t', header=0))
    test = rename_columns(pd.read_csv('/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/200/classification/yahoo_test.tsv', sep='\t', header=0))

    # train = train.dropna(subset=['text'])
    # train["text"] = train["text"].apply(lambda x: x.replace('"""',''))
    num_labels = len(set(train.iloc[:,1]))
    train["text"] = train["text"].apply(lambda x: str(x))
    label_set = list(set(train['label']))
    label_set.sort()

    label_map = {}
    for xx,lab in enumerate(label_set):
        label_map[lab] = xx

    train_labs = []
    for j in list(train['label']):
        train_labs.append(label_map[j])
    train['label'] = train_labs

    dev_labs = []
    for j in list(dev['label']):
        dev_labs.append(label_map[j])
    dev['label'] = dev_labs

    test_labs = []
    for j in list(test['label']):
        test_labs.append(label_map[j])
    test['label'] = test_labs
    train = datasets.Dataset.from_dict(train)
    dev = datasets.Dataset.from_dict(dev)
    test = datasets.Dataset.from_dict(test)

    dataset = datasets.DatasetDict({"train":train, "validation":dev, "test":test})
    # print(dataset)
    # print(dataset["train"].features["label"].feature.names)  # All label names
    # print(dataset["train"].features["label"].feature._int2str)  # Same as `.names`
    # print(dataset["train"].features["label"].feature._str2int)

    model_name = "bert-base-uncased"
    learning_rate=5e-5

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)


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
    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/200/classification/yahoo_200",
        save_steps = eval_steps,
        save_strategy='steps',
        save_total_limit = 1,
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

    # print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator = data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("/fs/nexus-projects/audio-visual_dereverberation/utkarsh_diff/diff/data/low_res/200/classification/yahoo_200")

    preds = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(preds[0], axis=-1)
    # print(preds, len(preds))

    print(f"F-1 Micro score : {metric.compute(predictions=preds, references=test['label'], average='micro')['f1']}")
    print(f"F-1 Macro score : {metric.compute(predictions=preds, references=test['label'], average='macro')['f1']}")

train_model()
