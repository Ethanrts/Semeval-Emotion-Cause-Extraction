import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Define the LSTM model with an additional linear layer
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)  # Additional linear layer
        self.fc2 = nn.Linear(hidden_dim // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Using the last output from the sequence
        out = self.fc1(lstm_out)
        out = self.dropout(out)
        output = self.fc2(out)
        return output

# Load the JSON dataset
with open('Subtask_1_train.json', 'r') as file:
    dataset = json.load(file)

# Extract features and labels
X = [utterance['text'] for conv in dataset for utterance in conv['conversation']]
y = [utterance['emotion'] for conv in dataset for utterance in conv['conversation']]

# Create vocabulary and encode text data
word_counts = Counter(" ".join(X).split())
word_to_index = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(), 1)}

# Convert text data to sequences of indices
X_encoded = [[word_to_index[word] for word in sentence.split()] for sentence in X]

# Pad sequences to a fixed length
max_seq_length = max(len(seq) for seq in X_encoded)
X_padded = [seq + [0] * (max_seq_length - len(seq)) for seq in X_encoded]

# Convert emotions to categorical labels
unique_emotions = list(set(y))
label_dict = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
y = [label_dict[emotion] for emotion in y]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Define hyperparameters
vocab_size = len(word_to_index) + 1
embedding_dim = 100
hidden_dim = 128
output_size = len(set(y))
num_layers = 1
dropout = 0.5
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Create DataLoader for training and testing
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# Initialize the LSTM model with an additional linear layer
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the LSTM model
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the LSTM model
model.eval()
predicted_labels = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')
class_report = classification_report(true_labels, predicted_labels)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", class_report)