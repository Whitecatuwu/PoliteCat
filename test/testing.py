import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 載入 tokenizer 與模型

import os

model_path = os.getcwd() + r"/test/model"  # 這裡是你 TrainingArguments 裡的 output_dir
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


def output_label(text) -> dict:

    # 2. 準備要預測的句子
    # text = "I'm sorry, but I must say that your appearance does not align with conventional standards of beauty."

    # 3. Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 4. 模型推論
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    # 5. 輸出結果
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    predictions = (probs > 0.5).int().squeeze().tolist()

    result: dict = {}
    for label, pred, prob in zip(labels, predictions, probs.squeeze().tolist()):
        result[label] = {"pred": bool(pred), "score": prob}
        # print(f"{label:15}: {'✅' if pred else '❌'} (score={prob:.4f})")

    return result


if __name__ == "__main__":
    # 測試輸出
    text = "You so ugly, you look like a monkey."
    result = output_label(text)
    for label, info in result.items():
        print(
            f"{label:15}: {'✅' if info['pred'] else '❌'} (score={info['score']:.4f})"
        )
