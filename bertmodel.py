import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load the JSON dataset
with open('Subtask_1_train.json', 'r') as file:
    dataset = json.load(file)

# Extract features and labels
X = [utterance['text'] for conv in dataset for utterance in conv['conversation']]
y = [utterance['emotion'] for conv in dataset for utterance in conv['conversation']]

# Convert emotions to categorical labels
unique_emotions = list(set(y))
label_dict = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
y = [label_dict[emotion] for emotion in y]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

# Tokenize and encode the text data
max_length = 128
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# Reduce the batch size
batch_size = 8  # Adjust according to available memory

# Check if CUDA is available and use it for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the available device
model.to(device)

# Move the training and testing data to the available device
train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)


# Fine-tune the BERT model
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):  # Number of epochs, adjust as needed
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_inputs = {key: val[i:i+batch_size] for key, val in train_encodings.items()}
        batch_labels = torch.tensor(y_train[i:i+batch_size])
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Epoch {epoch + 1}, Batch {i//batch_size + 1}, Loss: {running_loss / 100}")
            running_loss = 0.0

    print(f"Epoch {epoch + 1} completed")

# Evaluate the fine-tuned model
model.eval()
with torch.no_grad():
    test_outputs = model(**test_encodings)
    predicted = torch.argmax(test_outputs.logits, dim=1).tolist()

accuracy = accuracy_score(y_test, predicted)
f1 = f1_score(y_test, predicted, average='weighted')
class_report = classification_report(y_test, predicted)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", class_report)