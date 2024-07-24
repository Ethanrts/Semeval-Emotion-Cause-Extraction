import torch
import torch.nn as nn
import json
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Define a simple classifier on top of BERT with max pooling
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)  # BERT base model output size is 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        pooled_output = torch.max(outputs.last_hidden_state, dim=1)[0]  # Max pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    #     cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token output
    #     cls_output = self.dropout(cls_output)
    #     logits = self.fc(cls_output)
    #     return logits

with open('Subtask_1_train.json', 'r') as file:
    dataset = json.load(file)

X = []
y = []
count = 0
for conv in dataset:
  curr_conv = conv['conversation']
  cause_pairs = []
  for pair in conv['emotion-cause_pairs']:
    cause_pairs.append((pair[0][0], pair[1][0]))

  for i, utterance in enumerate(curr_conv):
    for j in range(i+1):

      curr = utterance['speaker'] + ': ' + utterance['text']

      emotion = utterance['emotion'] + ':'

      history = [utterance['speaker'] + ': ' + utterance['text'] for utterance in curr_conv[:j+1]]

      target_text = history[-1]
      context_text = history[:-1]

      # This concatenates strings to make the input look something like: "joy: chandler: me too ! [SEP] ross: i won a prize . [SEP]"
      transformed_input = [emotion] + [curr] + ["[SEP]"] + [target_text] + ["[SEP]"]

      # This will add context to the input
      # if context_text:
      #   transformed_input = transformed_input + [" ".join(context_text)]

      transformed_input = " ".join(transformed_input)

      X.append(transformed_input)

      target_utterance = curr_conv[j]

      if (str(utterance['utterance_ID']), str(target_utterance['utterance_ID'])) in cause_pairs:
        y.append(1)
      else:
        y.append(0)
  count += 1

len(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.3, test_size=0.1, random_state=42, stratify=y)

train_input_text = X_train
train_labels = y_train

val_input_text = X_val
val_labels = y_val

# Tokenize input
encoded_train_input = tokenizer(train_input_text, padding=True, truncation=True, return_tensors='pt').to(device)


# Prepare input tensors
input_ids = encoded_train_input['input_ids']
attention_mask = encoded_train_input['attention_mask']
labels = torch.tensor(train_labels).unsqueeze(1).to(device)  # Convert labels to tensor

# Create a DataLoader for mini-batching
dataset = TensorDataset(input_ids, attention_mask, labels)
batch_size = 32  # Adjust batch size according to your data
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import gc

# clear gpu cache
gc.collect()
torch.cuda.empty_cache()


# Instantiate the model
model = BertClassifier(bert_model, num_classes=1).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Adam optimizer

# Training loop with mini-batching
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        batch_input_ids, batch_attention_mask, batch_labels = batch
        outputs = model(batch_input_ids, batch_attention_mask)
        loss = criterion(outputs, batch_labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

# Free up memory to perform the training
del dataloader

# Prepare input tensors
encoded_val_input = tokenizer(val_input_text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
val_input_ids = encoded_val_input['input_ids']
val_attention_mask = encoded_val_input['attention_mask']
val_step_labels = torch.tensor(val_labels)

# Create a DataLoader for mini-batching
val_dataset = TensorDataset(val_input_ids, val_attention_mask)
batch_size = 16  # Adjust batch size according to your data
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model.eval()
all_predicted_labels = []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask = batch

        outputs = model(input_ids, attention_mask=attention_mask)

        probabilities = torch.sigmoid(outputs)
        predicted_labels = (probabilities > 0.5).long()

        all_predicted_labels.extend([label.item() for label in predicted_labels])

from sklearn.metrics import f1_score, recall_score

print(val_labels)
print(all_predicted_labels)
f1 = f1_score(val_step_labels, all_predicted_labels, average='binary')
print(f"F1 Score with max pooling and no context: {f1}")