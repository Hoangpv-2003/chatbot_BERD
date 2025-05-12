# train_bert_ner.py

import json
import numpy as np
from datasets import Dataset
from evaluate import load as load_metric

from transformers import (BertTokenizerFast, BertForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)

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
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# 4. Tokenize & align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'],
                                  truncation=True,
                                  is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore token (like [CLS], [SEP])
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                # Same word -> same label or I-label
                label_ids.append(label_to_id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

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
