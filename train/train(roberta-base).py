#%%
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. 載入資料
df = pd.read_csv("train.csv", encoding='utf-8', on_bad_lines='skip')
df = df[['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
df = df.fillna("")

# 2. Hugging Face Dataset & 分割
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# 3. Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["comment_text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 4. 處理 label
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def format_labels(example):
    example["labels"] = torch.tensor([example[col] for col in label_cols], dtype=torch.float)
    return example

train_dataset = train_dataset.map(format_labels)
val_dataset = val_dataset.map(format_labels)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


#%%
# 5. 模型初始化
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,
    problem_type="multi_label_classification"
)

# 6. 評估指標
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (preds > 0.5).astype(int)
    labels = p.label_ids
    return {
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
        'f1': f1_score(labels, preds, average='macro', zero_division=0),
        'accuracy': accuracy_score(labels, preds)
    }

# 7. 訓練參數
training_args = TrainingArguments(
    output_dir="results_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 9. 訓練
#%%
trainer.train()
trainer.save_model("results_roberta")

# %%
