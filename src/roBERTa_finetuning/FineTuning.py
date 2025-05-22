from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


df = pd.read_csv("iemocap_text_emotion_6labels.csv")

label_map = {'hap': 0, 'exc':0, # happy, excited
             'sad': 1, # sad
             'ang': 2, # angry
             'neu': 3, # neutral
             'fru':4, # frustrated
             'sur':5, 'fea':5, 'dis':5,'xxx':5, 'oth':5 # surprised, fearful, disgusted, indefinite, other
             }

# Etichette numeriche
#label2id = {label: i for i, label in enumerate(df['label'].unique())}
#df['label_id'] = df['label'].map(label2id)

# Suddividi in train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Convertili in Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Imposta formato PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Modifica per includere "labels" nel dataset
#train_dataset = train_dataset.rename_column("label_id", "labels")
#test_dataset = test_dataset.rename_column("label_id", "labels")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,
    id2label={v: k for k, v in label_map.items()},
    label2id=label_map
)

training_args = TrainingArguments(
    output_dir="roberta-iemocap-old",
    learning_rate=2e-5,
    eval_strategy="epoch",
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
    compute_metrics=compute_metrics
)

print("Start training")
trainer.train()
trainer.save_model("./roberta-iemocap/final-model")


predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=1)
true_labels = test_dataset["labels"]

labels = list(label_map.values())
target_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
print(classification_report(true_labels, pred_labels, labels=labels, target_names=target_names))


tokenizer.save_pretrained("./roberta-iemocap/final-model")
