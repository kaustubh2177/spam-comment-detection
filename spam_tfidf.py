import torch
import pandas as pd
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# Load dataset
data = pd.read_csv("youtube_spam.csv")
data = data[['CONTENT', 'CLASS']]
data.columns = ['text', 'label']

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_texts, train_labels)
val_dataset = SpamDataset(val_texts, val_labels)

model = AlbertForSequenceClassification.from_pretrained(
    "albert-base-v2",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

def predict(comment):
    inputs = tokenizer(comment, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Spam" if prediction == 1 else "Not Spam"

print(predict("Free shoes click link now"))
