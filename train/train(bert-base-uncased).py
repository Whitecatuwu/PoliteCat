import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# local_files_only=True 離線可用
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

# 用於確保在主模組中執行，當num_proc大於1時
# 如果你是在windows系統上運行，不要在 Jupyter Notebook 執行
if __name__ == "__main__":

    # 1. 載入資料
    df = pd.read_csv(os.getcwd() + r"/train/train.csv")
    df = df[["comment_text"] + label_cols]
    df = df.fillna("")
    print("載入資料完成，資料形狀:", df.shape)

    # 2. 資料集分割
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print("訓練集形狀:", train_dataset.shape, "驗證集形狀:", val_dataset.shape)

    # 3. Tokenizer
    def tokenize_function(example):
        return tokenizer(
            example["comment_text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
    )
    print("Tokenization 完成，訓練集和驗證集已轉換為 tokenized 格式。")

    # 4. 加入 labels
    def add_and_format_labels(example):
        """example["labels"] = torch.tensor(
            [example[col] for col in label_cols], dtype=torch.float
        )"""
        example["labels"] = [float(example[col]) for col in label_cols]
        return example

    train_dataset = train_dataset.map(
        add_and_format_labels, num_proc=os.cpu_count() // 2, load_from_cache_file=True
    )
    val_dataset = val_dataset.map(
        add_and_format_labels, num_proc=os.cpu_count() // 2, load_from_cache_file=True
    )
    print("Labels 已加入並轉換為 float tensor。")

    # 5. 設定格式
    # 使用 with_format 只在需要時轉成 tensor
    train_dataset = train_dataset.with_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset = val_dataset.with_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    print("資料集格式已設定為 torch tensor ，包含 input_ids, attention_mask, labels。")

    # 6. 模型初始化
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6, problem_type="multi_label_classification"
    )
    print("模型初始化完成，使用 BERT base uncased 模型，設定為多標籤分類。")

    # 7. 評估函數
    def compute_metrics(p):
        preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
        preds = (preds > 0.5).astype(int)
        labels = p.label_ids
        return {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
            "recall": recall_score(labels, preds, average="macro"),
            "precision": precision_score(labels, preds, average="macro"),
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
        dataloader_num_workers=os.cpu_count() // 2,
    )

    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 10. 開始訓練
    print("訓練參數設定完成，開始訓練...")
    trainer.train()
    tokenizer.save_pretrained("./results")
    print("訓練完成，模型和 tokenizer 已儲存到 ./results。")
