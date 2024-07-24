import json
import matplotlib.pyplot as plt

# Load the JSON file
with open('subtask_1_train.json', 'r') as file:
    data = json.load(file)

# Calculate sentence lengths
sentence_lengths = []
for conversation in data:
    for utterance in conversation['conversation']:
        sentence = utterance['text']
        # Split the sentence into words and count their number
        words = sentence.split()
        sentence_length = len(words)
        sentence_lengths.append(sentence_length)

# Create a histogram with bins for sentence lengths
plt.figure(figsize=(10, 6))
plt.hist(sentence_lengths, bins=range(min(sentence_lengths), max(sentence_lengths) + 2), color='skyblue', edgecolor='black')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length (Number of Words)')
plt.ylabel('Frequency')
plt.xticks(range(min(sentence_lengths), max(sentence_lengths) + 1))
plt.tight_layout()

# Show the histogram
plt.show()
