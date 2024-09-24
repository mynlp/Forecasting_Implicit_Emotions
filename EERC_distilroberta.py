import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import evaluate
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

model_ckpt = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, truncation_side='left')

# load data
emotions_train_valid = pd.read_csv("dailydialog/data_preproc.csv")
emotions_train_valid = emotions_train_valid[emotions_train_valid["split"]!="test"][["split", "history", "utterance", "emotion"]]
emotions_test = pd.read_csv("dailydialog/data_preproc_next_emotion.csv")
emotions_test = emotions_test[emotions_test["split"]=="test"][["split", "history", "next_utterance", "next_uttr_emotion"]]

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 3
id2label = {
    "0": "positive",
    "1": "neutral",
    "2": "negative"
}
label2id = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1score = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    prec = precision.compute(predictions=predictions, references=labels, average='macro', zero_division=0.0)["precision"]
    rec = recall.compute(predictions=predictions, references=labels, average='macro')["recall"]
    f1 = f1score.compute(predictions=predictions, references=labels, average='macro')["f1"]
    predictions2 = [predictions[i] for i, label in enumerate(labels) if label != 1]
    labels2 = [label for label in labels if label != 1]
    f1_2 = f1score.compute(predictions=predictions2, references=labels2, average='macro')["f1"]
    f1_3 = f1score.compute(predictions=predictions, references=labels, labels=[0, 2], average='macro')["f1"]
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "f1_wo_neu_old": f1_2, "f1_wo_neu": f1_3}

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def make_dataset(mode):
    if mode not in set(["full-history", "last-uttr", "no-history"]):
        print("mode is invalid")
        return
    if mode == "full-history":
        df_train_valid = deepcopy(emotions_train_valid)
        df_train_valid["history"] = df_train_valid["history"].apply(lambda x: " ".join(eval(x)))
        df_test = deepcopy(emotions_test)
        df_test["history"] = df_test["history"].apply(lambda x: " ".join(eval(x))) + " " + df_test["next_utterance"]
        df = pd.concat([df_train_valid[["split", "history", "emotion"]],
                        df_test[["split", "history", "next_uttr_emotion"]].rename(columns={"next_uttr_emotion": "emotion"})], ignore_index=True)
        df.rename(columns={"history": "text", "emotion": "label"}, inplace=True)
    elif mode == "last-uttr":
        df_train_valid = deepcopy(emotions_train_valid)
        df_train_valid["history"] = df_train_valid["history"].apply(lambda x: " ".join(eval(x)[-2:]) if len(eval(x)) >= 2 else eval(x)[-1])
        df_test = deepcopy(emotions_test)
        df_test["history"] = df_test["history"].apply(lambda x: eval(x)[-1]) + " " + df_test["next_utterance"]
        df = pd.concat([df_train_valid[["split", "history", "emotion"]],
                        df_test[["split", "history", "next_uttr_emotion"]].rename(columns={"next_uttr_emotion": "emotion"})], ignore_index=True)
        df.rename(columns={"history": "text", "emotion": "label"}, inplace=True)
    else:
        df_train_valid = deepcopy(emotions_train_valid)
        df_test = deepcopy(emotions_test)
        df = pd.concat([df_train_valid[["split", "utterance", "emotion"]],
                        df_test[["split", "next_utterance", "next_uttr_emotion"]].rename(columns={"next_utterance": "utterance", "next_uttr_emotion": "emotion"})], ignore_index=True)
        df.rename(columns={"utterance": "text", "emotion": "label"}, inplace=True)
        
    df["label"].replace({"positive": 0, "neutral": 1, "negative": 2}, inplace=True)
    df_train = df[df["split"]=="train"]
    df_valid = df[df["split"]=="valid"]
    df_test = df[df["split"]=="test"]
    label_ratio = df_train.value_counts("label", sort=False, normalize=True)
    emotions_dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train[["text", "label"]], preserve_index=False).shuffle(),
        "valid": Dataset.from_pandas(df_valid[["text", "label"]], preserve_index=False),
        "test": Dataset.from_pandas(df_test[["text", "label"]], preserve_index=False)
    })
    return label_ratio, emotions_dataset

for mode in ["full-history", "last-uttr", "no-history"]:
    print(mode)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)
    label_ratio, emotions_dataset = make_dataset(mode)
    emotions_encoded = emotions_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=f"EERC-{model_ckpt}-{mode}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=40,
        weight_decay=0.01,
        warmup_steps=500,
        metric_for_best_model="f1",
        load_best_model_at_end=True
    )

    class_weight = torch.tensor(1/label_ratio).clone().to(device, torch.float32)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=class_weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=emotions_encoded["train"],
        eval_dataset=emotions_encoded["valid"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()
    trainer.save_model(f"EERC-{model_ckpt}-{mode}")
    
    print("Results on Test Data:")
    preds_output = trainer.predict(emotions_encoded["test"])
    print(preds_output.metrics)
