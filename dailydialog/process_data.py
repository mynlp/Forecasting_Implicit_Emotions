import pandas as pd

with open("train/dialogues_train.txt") as f:
    train_text = f.readlines()
with open("train/dialogues_emotion_train.txt") as f:
    train_emo = f.readlines()
with open("validation/dialogues_validation.txt") as f:
    valid_text = f.readlines()
with open("validation/dialogues_emotion_validation.txt") as f:
    valid_emo = f.readlines()
with open("test/dialogues_test.txt") as f:
    test_text = f.readlines()
with open("test/dialogues_emotion_test.txt") as f:
    test_emo = f.readlines()

train_text = [[sen.strip() for sen in text.split('__eou__')[:-1]] for text in train_text]
train_emo = [emo.split(' ')[:-1] for emo in train_emo]
valid_text = [[sen.strip() for sen in text.split('__eou__')[:-1]] for text in valid_text]
valid_emo = [emo.split(' ')[:-1] for emo in valid_emo]
test_text = [[sen.strip() for sen in text.split('__eou__')[:-1]] for text in test_text]
test_emo = [emo.split(' ')[:-1] for emo in test_emo]

idto3 = {
    "0": "neutral",
    "1": "negative",
    "2": "negative",
    "3": "negative",
    "4": "positive",
    "5": "negative"
}

def preprocess(data_text, data_emo, split):
    output = []
    for texts, emos in zip(data_text, data_emo):
        history = []
        for utt, emo in zip(texts, emos):
            history.append(utt)
            if emo != "6":
                output.append([split, history.copy(), utt, idto3[emo]])
    return pd.DataFrame(output, columns=["split", "history", "utterance", "emotion"])

df_train = preprocess(train_text, train_emo, "train")
df_valid = preprocess(valid_text, valid_emo, "valid")
df_test = preprocess(test_text, test_emo, "test")

print(len(df_train), len(df_valid), len(df_test)) # 85570 7962 7624
pd.concat([df_train, df_valid, df_test]).to_csv('data_preproc.csv', index=False)
