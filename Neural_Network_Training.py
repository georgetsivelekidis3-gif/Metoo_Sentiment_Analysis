import numpy as np
import pandas as pd
import torch
import emoji
from sklearn.metrics import classification_report
from transformers import BertTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Label mappings
sentiment_mapping = {'NEGATIVE': 0, 'POSITIVE': 1, 'NEUTRAL': 2}
emotion_mapping = {'ANGER': 0, 'FEAR': 1, 'JOY': 2, 'SADNESS': 3, 'SURPRISE': 4, 'DISGUST': 5, 'NO-EMOTION': 6}

# Convert emojis to text
def convert_emojis_to_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))  # Spaces

# Load data from CSV
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    tweets = df.iloc[:, 0].astype(str).apply(convert_emojis_to_text).values  # Apply emoji conversion
    sentiments = df.iloc[:, 1].astype(str).map(lambda x: sentiment_mapping.get(x.strip(), -1)).values
    emotions = df.iloc[:, 2].astype(str).map(lambda x: emotion_mapping.get(x.strip(), -1)).values
    
    valid_indices = np.where((sentiments != -1) & (emotions != -1))[0]
    return tweets[valid_indices], sentiments[valid_indices], emotions[valid_indices]

# Load datasets
tweets_2023_train, sentiments_2023_train, emotions_2023_train = load_csv_data("Metoo2023-Training.csv")
tweets_2023_test, sentiments_2023_test, emotions_2023_test = load_csv_data("Metoo2023-Test.csv")

tweets_2024_train, sentiments_2024_train, emotions_2024_train = load_csv_data("Metoo2024-Training.csv")
tweets_2024_test, sentiments_2024_test, emotions_2024_test = load_csv_data("Metoo2024-Test.csv")

# Tokenizer initialization
tokenizer = BertTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# Define dataset class
class GreekBERTDataset(Dataset):
    def __init__(self, texts, sentiments, emotions, tokenizer, max_len=64):  
        self.texts = texts
        self.sentiments = sentiments
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_labels': torch.tensor(self.sentiments[idx], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotions[idx], dtype=torch.long)
        }

# Prepare datasets
train_dataset_2023 = GreekBERTDataset(tweets_2023_train, sentiments_2023_train, emotions_2023_train, tokenizer)
test_dataset_2023 = GreekBERTDataset(tweets_2023_test, sentiments_2023_test, emotions_2023_test, tokenizer)

train_dataset_2024 = GreekBERTDataset(tweets_2024_train, sentiments_2024_train, emotions_2024_train, tokenizer)
test_dataset_2024 = GreekBERTDataset(tweets_2024_test, sentiments_2024_test, emotions_2024_test, tokenizer)

train_dataset_combined = GreekBERTDataset(
    np.concatenate([tweets_2023_train, tweets_2024_train]),
    np.concatenate([sentiments_2023_train, sentiments_2024_train]),
    np.concatenate([emotions_2023_train, emotions_2024_train]),
    tokenizer
)

test_dataset_combined = GreekBERTDataset(
    np.concatenate([tweets_2023_test, tweets_2024_test]),
    np.concatenate([sentiments_2023_test, sentiments_2024_test]),
    np.concatenate([emotions_2023_test, emotions_2024_test]),
    tokenizer
)

# Model definition
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class MultiTaskBERT(BertPreTrainedModel):
    _tied_weights_keys = []
    all_tied_weights_keys = []

    def __init__(self, config, num_sentiment_labels=3, num_emotion_labels=7):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)
        self.emotion_classifier = nn.Linear(config.hidden_size, num_emotion_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask, sentiment_labels=None, emotion_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        
        loss = None
        if sentiment_labels is not None and emotion_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            emotion_loss = loss_fct(emotion_logits, emotion_labels)
            loss = sentiment_loss + emotion_loss
        
        return {'loss': loss, 'sentiment_logits': sentiment_logits, 'emotion_logits': emotion_logits}

# Initialize model
model = MultiTaskBERT.from_pretrained("nlpaueb/bert-base-greek-uncased-v1").to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

def train_and_evaluate(train_dataset, test_dataset, dataset_name):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    y_sent_pred = np.argmax(predictions.predictions[0], axis=-1)
    y_emo_pred = np.argmax(predictions.predictions[1], axis=-1)
    y_sent_true = np.array([test_dataset[i]['sentiment_labels'].item() for i in range(len(test_dataset))])
    y_emo_true = np.array([test_dataset[i]['emotion_labels'].item() for i in range(len(test_dataset))])
    print(f"{dataset_name} Classification Report:")
    print(classification_report(y_sent_true, y_sent_pred, target_names=list(sentiment_mapping.keys())))
    print(classification_report(y_emo_true, y_emo_pred, target_names=list(emotion_mapping.keys())))

print("Training on Metoo2023...")
train_and_evaluate(train_dataset_2023, test_dataset_2023, "Metoo2023")

print("Training on Metoo2024...")
train_and_evaluate(train_dataset_2024, test_dataset_2024, "Metoo2024")

print("Training on Combined Dataset...")
train_and_evaluate(train_dataset_combined, test_dataset_combined, "Combined Dataset")
