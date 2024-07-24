from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

class EmotionDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        encoding = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load the JSON dataset
with open('/content/Subtask_1_train.json', 'r') as file:
    dataset = json.load(file)

# Extract features and labels
X = [utterance['text'] for conv in dataset for utterance in conv['conversation']]
y = [utterance['emotion'] for conv in dataset for utterance in conv['conversation']]

# Convert emotions to categorical labels
unique_emotions = list(set(y))
label_dict = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
y = [label_dict[emotion] for emotion in y]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_filler, X_train_subset, y_filler, y_train_subset = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define BERT model pretrained weights for tokenizer and model
model_name = 'bert-base-uncased'

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize and encode the text data
max_length = 128
train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_length)
train_subset_dataset = EmotionDataset(X_train_subset, y_train_subset, tokenizer, max_length)
test_dataset = EmotionDataset(X_test, y_test, tokenizer, max_length)

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_subset_dataloader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check if CUDA is available and use it for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define BERT configuration
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
hidden_act = "gelu"
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
max_position_embeddings = 512
initializer_range = 0.02
layer_norm_eps = 1e-12

configuration = BertConfig(
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    hidden_act=hidden_act,
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    max_position_embeddings=max_position_embeddings,
    initializer_range=initializer_range,
    layer_norm_eps=layer_norm_eps,
    num_labels=len(set(y))
)

# Initialize BERT model
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=configuration)
model.to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    count = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    print(f"Epoch {epoch + 1} completed")
    print(running_loss / count)

# Evaluation on the training subset
model.eval()
predicted_train_subset = []
true_train_subset = []

with torch.no_grad():
    for batch in train_subset_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predicted = torch.argmax(outputs.logits, dim=1).tolist()

        predicted_train_subset.extend(predicted)
        true_train_subset.extend(labels.tolist())

accuracy_train_subset = accuracy_score(true_train_subset, predicted_train_subset)
f1_train_subset = f1_score(true_train_subset, predicted_train_subset, average='weighted')
class_report_train_subset = classification_report(true_train_subset, predicted_train_subset)

print("Evaluation on Training Subset:")
print(f"Accuracy: {accuracy_train_subset}")
print(f"F1 Score: {f1_train_subset}")
print("Classification Report:\n", class_report_train_subset)

# Evaluation on the test set
model.eval()
predicted_test = []
true_test = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predicted = torch.argmax(outputs.logits, dim=1).tolist()

        predicted_test.extend(predicted)
        true_test.extend(labels.tolist())

accuracy_test = accuracy_score(true_test, predicted_test)
f1_test = f1_score(true_test, predicted_test, average='weighted')
class_report_test = classification_report(true_test, predicted_test)

print("Evaluation on Test Set:")
print(f"Accuracy: {accuracy_test}")
print(f"F1 Score: {f1_test}")
print("Classification Report:\n", class_report_test)
