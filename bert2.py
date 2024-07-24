import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import gc

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_filler, X_train_subset, y_filler, y_train_subset = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define BERT model pretrained weights for tokenizer and model
model_name = 'bert-base-uncased'

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize and encode the text data
max_length = 128
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
train_subset_encodings = tokenizer(X_train_subset, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# Check if CUDA is available and use it for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the training and testing data to the available device
train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
train_subset_encodings = {key: val.to(device) for key, val in train_subset_encodings.items()}
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

# Hyperparameters

# Defaults:
# hidden_size = 768
# num_hidden_layers = 12
# num_attention_heads = 12
# intermediate_size = 3072
# hidden_act = "gelu"
# hidden_dropout_prob = 0.1
# attention_probs_dropout_prob = 0.1
# max_position_embeddings = 512
# initializer_range = 0.02
# layer_norm_eps = 1e-12

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

num_epochs = 3
batch_size = 8

configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob, max_position_embeddings=max_position_embeddings, initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, num_labels=len(set(y)))

model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_name, config = configuration)

model.to(device)

# clear gpu cache
gc.collect()
torch.cuda.empty_cache()

# Training, reports average loss on each epoch
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(num_epochs):  # Number of epochs, adjust as needed
    running_loss = 0.0
    count = 0
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_inputs = {key: val[i:i+batch_size] for key, val in train_encodings.items()}
        batch_labels = torch.tensor(y_train[i:i+batch_size]).to(device)
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    print(f"Epoch {epoch + 1} completed")
    print(running_loss / count)

# dropout 0.1, epochs 3
# Evaluate the fit to the training set
model.eval()
with torch.no_grad():
    train_outputs = model(**train_subset_encodings)
    predicted = torch.argmax(train_outputs.logits, dim=1).tolist()

accuracy = accuracy_score(y_train_subset, predicted)
f1 = f1_score(y_train_subset, predicted, average='weighted')
class_report = classification_report(y_train_subset, predicted)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", class_report)

# dropout 0.1, epochs 3
# Evaluate on the validation set
model.eval()
predicted_emotions = []

with torch.no_grad():
    test_outputs = model(**test_encodings)
    predicted = torch.argmax(test_outputs.logits, dim=1).tolist()

    for idx, pred_label in enumerate(predicted):
        predicted_emotions.append({
            "text": X_test[idx],
            "predicted_emotion": unique_emotions[pred_label]
        })

    # Saving the predicted emotions to a JSON file
    with open('predicted_emotions.json', 'w') as outfile:
        json.dump(predicted_emotions, outfile, indent=4)

# Rest of the evaluation code remains the same
accuracy = accuracy_score(y_test, predicted)
f1 = f1_score(y_test, predicted, average='weighted')
class_report = classification_report(y_test, predicted)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Classification Report:\n", class_report)