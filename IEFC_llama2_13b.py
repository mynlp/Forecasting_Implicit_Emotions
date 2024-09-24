import pandas as pd
from transformers import LlamaTokenizer, LlamaForSequenceClassification, BitsAndBytesConfig
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
import numpy as np
from tqdm import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter
import os
from copy import deepcopy

login(token="")

# load data
primaryto3 = {
    "anticipation": "positive",
    "joy": "positive",
    "trust": "positive",
    "surprise": "neutral",
    "neutral": "neutral",
    "fear": "negative",
    "sadness": "negative",
    "disgust": "negative",
    "anger": "negative"
}
emotions_train_valid = pd.read_csv("memor/data_preproc_2.csv").replace({'elicited_emotion': primaryto3})
emotions_train_valid = emotions_train_valid[emotions_train_valid["split"]!="test"][["split", "history", "utterance", "elicited_emotion"]]
emotions_test = pd.read_csv("memor/test_data.csv").replace({'elicited_emotion': primaryto3})
emotions_test = emotions_test[["split", "history", "utterance", "elicited_emotion"]]
emotions = pd.concat([emotions_train_valid, emotions_test], ignore_index=True)

class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return np.array([idx])

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_ckpt = "meta-llama/Llama-2-13b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_ckpt)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

tokenizer.pad_token = tokenizer.eos_token

epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3, average="macro")
recall = Recall(task="multiclass", num_classes=3, average="macro")
f1 = F1Score(task="multiclass", num_classes=3, average="macro")
f1_2 = evaluate.load("f1")

def make_dataset(mode):
    if mode not in set(["full-history", "last-uttr", "no-history"]):
        print("mode is invalid")
        return
    if mode == "full-history":
        df = deepcopy(emotions)
        df["history"] = df["history"].apply(lambda x: " ".join(eval(x)))
        df.rename(columns={"history": "text", "elicited_emotion": "label"}, inplace=True)
    elif mode == "last-uttr":
        df = deepcopy(emotions)
        df["history"] = df["history"].apply(lambda x: " ".join(eval(x)[-2:]) if len(eval(x)) >= 2 else eval(x)[-1])
        df.rename(columns={"history": "text", "elicited_emotion": "label"}, inplace=True)
    else:
        df = deepcopy(emotions)
        df.rename(columns={"utterance": "text", "elicited_emotion": "label"}, inplace=True)
        
    df["label"].replace({"positive": 0, "neutral": 1, "negative": 2}, inplace=True)
    df_train = df[df["split"]=="train"]
    df_valid = df[df["split"]=="valid"]
    df_test = df[df["split"]=="test"]
    label_ratio = df_train.value_counts("label", sort=False, normalize=True)
    return label_ratio, df_train, df_valid, df_test

for mode in ["full-history", "last-uttr", "no-history"]:
    for lr in [1e-5, 2e-5, 4e-5]:
        for bsz in [1, 2, 4]:
            model = LlamaForSequenceClassification.from_pretrained(
                model_ckpt,
                device_map="auto",
                quantization_config=bnb_config,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            peft_config = LoraConfig(
                r=16,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=find_all_linear_names(model),
                bias="none",
                modules_to_save=["classifier"]
            )
            model = get_peft_model(model, peft_config)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.print_trainable_parameters()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
            writer = SummaryWriter()
            dirname = f"IEFC-{model_ckpt[11:]}-{mode}-{lr}-{bsz}"
            os.makedirs(dirname, exist_ok=True)
            model_path = f"{dirname}/model_weight.pth"
    
            label_ratio, df_train, df_valid, df_test = make_dataset(mode)
            # print(df_train.loc[0, "text"])
            # print(df_train.loc[1, "text"])
            # print(df_train.loc[2, "text"])
            train_data = CustomDataset(df_train)
            valid_data = CustomDataset(df_valid)
            test_data = CustomDataset(df_test)
            train_dataloader = DataLoader(train_data, batch_size=bsz, shuffle=True)
            valid_dataloader = DataLoader(valid_data, batch_size=4)
            test_dataloader = DataLoader(test_data, batch_size=2)
    
            class_weight = torch.tensor(1 / label_ratio).clone().to(device, torch.float16)
            loss_fn = nn.CrossEntropyLoss(weight=class_weight)
            best_valF1 = 0.0
    
            for epoch in range(epochs):
                print("epochs: ", epoch)
    
                # Train
                total_train_loss = 0.0
                predLs = []
                labelLs = []
                for batchIdx, sampledIdx in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                    sampledIdx = sampledIdx.cpu().data.numpy()
                    model.train()
                    optimizer.zero_grad()
    
                    sampledRowText = list(df_train["text"].iloc[list(sampledIdx.flatten())])
                    sampledRowLabels = torch.tensor(list(df_train["label"].iloc[list(sampledIdx.flatten())])).to(
                        device)
                    encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, return_tensors='pt').to(
                        device)  # Output shape: [bs, num_Labels]
                    encoded_inputIds = encoded_input["input_ids"].to(device)
                    encoded_attnMask = encoded_input["attention_mask"].to(device)
                    outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
                    logits = outputs.logits
                    loss = loss_fn(logits, sampledRowLabels)
                    loss.backward()
                    optimizer.step()
                    predLs.append(torch.argmax(logits, dim=1).flatten().cpu().data.numpy())
                    labelLs.append(sampledRowLabels.cpu().data.numpy())
                    total_train_loss += loss.item()
    
                writer.add_scalar('Loss/train', total_train_loss / len(train_dataloader), epoch)
                predLs = torch.tensor(np.concatenate(predLs))
                labelLs = torch.tensor(np.concatenate(labelLs))
                trainAcc = float(accuracy(predLs, labelLs))
                print("train accuracy: ", trainAcc)
                writer.add_scalar('Acc/train', trainAcc, epoch)
                trainPrec = float(precision(predLs, labelLs))
                print("train precision: ", trainPrec)
                writer.add_scalar('Prec/train', trainPrec, epoch)
                trainRec = float(recall(predLs, labelLs))
                print("train recall: ", trainRec)
                writer.add_scalar('Rec/train', trainRec, epoch)
                trainF1 = float(f1(predLs, labelLs))
                print("train f value: ", trainF1)
                writer.add_scalar('F1/train', trainF1, epoch)
    
                # Validation
                total_val_loss = 0.0
                predLs = []
                labelLs = []
                for batchIdx, sampledIdx in enumerate(valid_dataloader):
                    model.eval()
    
                    sampledRowText = list(df_valid["text"].iloc[list(sampledIdx.flatten())])
                    sampledRowLabels = torch.tensor(list(df_valid["label"].iloc[list(sampledIdx.flatten())])).to(
                        device)
                    encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, return_tensors='pt').to(
                        device)  # Output shape: [bs, num_Labels]
                    encoded_inputIds = encoded_input["input_ids"].to(device)
                    encoded_attnMask = encoded_input["attention_mask"].to(device)
                    outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
                    logits = outputs.logits
                    loss = loss_fn(logits, sampledRowLabels)
                    predLs.append(torch.argmax(logits, dim=1).flatten().cpu().data.numpy())
                    labelLs.append(sampledRowLabels.cpu().data.numpy())
                    total_val_loss += loss.item()
    
                writer.add_scalar('Loss/valid', total_val_loss / len(valid_dataloader), epoch)
                predLs = torch.tensor(np.concatenate(predLs))
                labelLs = torch.tensor(np.concatenate(labelLs))
                valAcc = float(accuracy(predLs, labelLs))
                print("validation accuracy: ", valAcc)
                writer.add_scalar('Acc/valid', valAcc, epoch)
                valPrec = float(precision(predLs, labelLs))
                print("validation precision: ", valPrec)
                writer.add_scalar('Prec/valid', valPrec, epoch)
                valRec = float(recall(predLs, labelLs))
                print("validation recall: ", valRec)
                writer.add_scalar('Rec/valid', valRec, epoch)
                valF1 = float(f1(predLs, labelLs))
                print("validation f value: ", valF1)
                writer.add_scalar('F1/valid', valF1, epoch)
                if best_valF1 < valF1:
                    best_valF1 = valF1
                    torch.save(model.state_dict(), model_path)
    
            writer.close()
    
            # Test
            model.load_state_dict(torch.load(model_path))
            predLs = []
            labelLs = []
            for batchIdx, sampledIdx in enumerate(test_dataloader):
                model.eval()
    
                sampledRowText = list(df_test["text"].iloc[list(sampledIdx.flatten())])
                sampledRowLabels = torch.tensor(list(df_test["label"].iloc[list(sampledIdx.flatten())]))
                encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, return_tensors='pt').to(
                    device)  # Output shape: [bs, num_Labels]
                encoded_inputIds = encoded_input["input_ids"].to(device)
                encoded_attnMask = encoded_input["attention_mask"].to(device)
                outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
                logits = outputs.logits
                predLs.append(torch.argmax(logits, dim=1).flatten().cpu().data.numpy())
                labelLs.append(sampledRowLabels.cpu().data.numpy())
    
            predLs = torch.tensor(np.concatenate(predLs))
            labelLs = torch.tensor(np.concatenate(labelLs))
            testAcc = float(accuracy(predLs, labelLs))
            print("test accuracy: ", testAcc)
            testPrec = float(precision(predLs, labelLs))
            print("test precision: ", testPrec)
            testRec = float(recall(predLs, labelLs))
            print("test recall: ", testRec)
            testF1 = float(f1(predLs, labelLs))
            print("test f value: ", testF1)
            testF1_2 = f1_2.compute(predictions=predLs, references=labelLs, labels=[0, 2], average='macro')["f1"]
            print("test f without neutral value: ", testF1_2)
