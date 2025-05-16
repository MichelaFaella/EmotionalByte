from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from datasets import Dataset

# Carica i dati
df = pd.read_csv("iemocap_text_emotion.csv")  # Assicurati che abbia 'text' e 'label'

# Etichette numeriche
label2id = {label: i for i, label in enumerate(df['label'].unique())}
df['label_id'] = df['label'].map(label2id)

# Suddividi in train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'])

# Convertili in Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Imposta formato PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

# Modifica per includere "labels" nel dataset
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label2id),
    id2label={v: k for k, v in label2id.items()},
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./roberta-iemocap",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model("./roberta-iemocap/final-model")


predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=1)
true_labels = test_dataset["labels"]

from sklearn.metrics import classification_report
print(classification_report(true_labels, pred_labels, target_names=label2id.keys()))

tokenizer.save_pretrained("./roberta-iemocap/final-model")
