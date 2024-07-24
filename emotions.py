import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the JSON dataset
with open('Subtask_1_train.json', 'r') as file:
    dataset = json.load(file)

# Extract features and labels
X = [utterance['text'] for conv in dataset for utterance in conv['conversation']]
y = [utterance['emotion'] for conv in dataset for utterance in conv['conversation']]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm = SVC(kernel='linear')
svm.fit(X_train_tfidf, y_train)

# Evaluate the model
accuracy = svm.score(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy}")

# Predict emotions for test set
y_pred = svm.predict(X_test_tfidf)

# Classification report
print(classification_report(y_test, y_pred))
