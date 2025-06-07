import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score

# 1. 載入資料
df = pd.read_csv("train.csv")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[['comment_text'] + label_cols]
df = df.fillna("")


dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# 3. Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["comment_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 4. 加入 labels 並轉成 float tensor
def add_and_format_labels(example):
    example["labels"] = torch.tensor([example[col] for col in label_cols], dtype=torch.float)
    return example

train_dataset = train_dataset.map(add_and_format_labels)
val_dataset = val_dataset.map(add_and_format_labels)

# 5. 設定格式
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 6. 模型初始化
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6,
    problem_type="multi_label_classification"
)

# 7. 評估函數
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (preds > 0.5).astype(int)
    labels = p.label_ids
    return {
        'f1_macro': f1_score(labels, preds, average='macro'),
        'accuracy': accuracy_score(labels, preds)
    }

# 8. 訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. 開始訓練
trainer.train()
tokenizer.save_pretrained("./results")


