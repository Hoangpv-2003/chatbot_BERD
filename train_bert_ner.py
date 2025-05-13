# train_bert_ner.py

import json
import numpy as np
from datasets import Dataset
from evaluate import load as load_metric

from transformers import (DataCollatorForTokenClassification, TrainingArguments, Trainer)
from transformers import PhobertTokenizer
from transformers import RobertaForTokenClassification, AutoTokenizer

model_name = "vinai/phobert-base"  # Hoặc tên mô hình bạn đang sử dụng
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


# 1. Load dataset
def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

train_dataset = load_dataset('data/train.json')
val_dataset = load_dataset('data/val.json')

# 2. Define labels
label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# 3. Load tokenizer & model

model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))


# 4. Tokenize & align labels
def tokenize_and_align_labels(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for tokens, labels in zip(examples['tokens'], examples['labels']):
        # Token hóa từng câu/tập từ
        encoding = tokenizer(tokens,
                             is_split_into_words=True,
                             truncation=True,
                             padding='max_length',
                             max_length=128,  # có thể điều chỉnh theo mô hình
                             return_attention_mask=True,
                             return_tensors='pt')

        input_ids = encoding['input_ids'][0].tolist()
        attention_mask = encoding['attention_mask'][0].tolist()

        # Gán nhãn: label từng từ, sau đó gán -100 cho các token vượt quá
        label_ids = [label_to_id.get(lbl, label_to_id["O"]) for lbl in labels]
        label_ids = label_ids + [-100] * (len(input_ids) - len(label_ids))
        label_ids = label_ids[:len(input_ids)]  # cắt lại đúng độ dài input

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(label_ids)

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }



train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# 5. Data collator (tự padding)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 6. Evaluation metrics (F1-score)
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./bert_ner_model",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9. Train
trainer.train()

# 10. Save final model
trainer.save_model("./bert_ner_model")
tokenizer.save_pretrained("./bert_ner_model")
