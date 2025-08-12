import math
from collections import defaultdict

# Training data
data = [
    ("Buy cheap now", "spam"),
    ("Limited offer, buy now", "spam"),
    ("Hey, how are you?", "ham"),
    ("Let’s catch up tomorrow", "ham"),
    ("Win cash prizes", "spam"),
    ("Meeting at 5pm", "ham")
]

# Train the model
def train(data):
    word_counts = {"spam": defaultdict(int), "ham": defaultdict(int)}
    class_counts = {"spam": 0, "ham": 0}
    total_words = {"spam": 0, "ham": 0}

    for text, label in data:
        class_counts[label] += 1
        for word in text.lower().split():
            word_counts[label][word] += 1
            total_words[label] += 1

    return word_counts, class_counts, total_words

# Predict label
def predict(text, word_counts, class_counts, total_words):
    words = text.lower().split()
    vocab = set(w for c in word_counts for w in word_counts[c])
    total_docs = sum(class_counts.values())
    scores = {}

    for label in class_counts:
        log_prob = math.log(class_counts[label] / total_docs)
        for word in words:
            freq = word_counts[label][word]
            prob = (freq + 1) / (total_words[label] + len(vocab))
            log_prob += math.log(prob)
        scores[label] = log_prob

    return max(scores, key=scores.get)

# Run
word_counts, class_counts, total_words = train(data)
tests = ["Buy now and win", "Meeting soon?", "You won a prize"]

for t in tests:
    print(f"'{t}' => {predict(t, word_counts, class_counts, total_words)}")
