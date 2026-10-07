import math
from collections import defaultdict

"""
================================================================================
THE MATHEMATICAL DERIVATION OF THE LOG-NAIVE BAYES FORMULA
================================================================================

This method combines Bayes' Theorem, the Naive Bayes assumption, and a
logarithmic trick to calculate classification scores without crashing.

Step 1: Start with Bayes' Theorem
----------------------------------
We want to find the probability that a document belongs to a specific category
(Class), given the words inside it:
    P(Class | Words) = [ P(Class) * P(Words | Class) ] / P(Words)

Because the denominator P(Words) is identical for every single class we test,
we can completely ignore it. We just need to find which class gives the highest
numerator:
    Score = P(Class) * P(Words | Class)

Step 2: The "Naive" Assumption
-------------------------------
A document is a collection of individual words (w1, w2, w3...). To calculate
P(Words | Class), the model makes a "naive" assumption: it assumes every word
appears completely independently of every other word.

Because of this assumption, we can calculate the total probability of the
document by simply multiplying the probabilities of each individual word together:
    Score = P(Class) * P(w1 | Class) * P(w2 | Class) * P(w3 | Class) * ...

Step 3: The Problem (Floating-Point Underflow)
-----------------------------------------------
Imagine a real-world email with 100 words. The probability of any single word
appearing in a spam email might be tiny, like 0.005. If you multiply 100 tiny
decimals together:
    0.005 * 0.002 * 0.0001 * ...
The result becomes an incredibly small number (e.g., 10^-150). Computers cannot
handle numbers this small and will round them down to absolute 0.0. Once your
score hits zero, the model can no longer tell which class is better.

Step 4: The Logarithm Trick to the Rescue
------------------------------------------
To fix this, we apply a mathematical property of logarithms. Logs turn
multiplication into addition:
    log(A * B) = log(A) + log(B)

If we apply log to our entire Naive Bayes formula, the multiplication signs
instantly turn into addition signs:
    log(Score) = log(P(Class)) + log(P(w1 | Class)) + log(P(w2 | Class)) + ...

Because logarithms preserve order (if X > Y, then log(X) > log(Y)), the category
that gets the highest log score is guaranteed to be the exact same category
that would have won using standard multiplication.

--------------------------------------------------------------------------------
DIRECT MAPPING TO THE PYTHON CODE:
--------------------------------------------------------------------------------
* log(P(Class))              -> score = math.log(self.class_counts[cls] / total_docs)
* +                          -> score += ...
* log(P(w | Class))          -> math.log((count + self.smoothing) /
                                         (total + self.smoothing * vocab_size))
================================================================================
"""

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)

            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}

    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)

    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")

if __name__ == "__main__":
    # Train on spam data
    train_docs = [
        "win free money now",
        "free lottery ticket winner",
        "claim your prize today free",
        "urgent offer free cash",
        "congratulations you won free",
        "meeting tomorrow at noon",
        "project update attached",
        "can we schedule a call",
        "quarterly report review",
        "lunch on thursday sounds good",
        "team standup notes attached",
        "please review the pull request",
    ]

    train_labels = [
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
    ]

    classifier = NaiveBayes()
    classifier.train(train_docs, train_labels)

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]

    for msg in test_messages:
        print(f"  '{msg}' -> {classifier.predict(msg)}")

    # Inspect the learned probabilities
    print("\nTop spam words:")
    show_top_words(classifier, "spam")
    print("\nTop ham words:")
    show_top_words(classifier, "ham")

# Phase 1: Initialization (classifier = NaiveBayes())
# When this line runs, the model creates empty storage structures:
# • self.smoothing = 1.0
# • self.class_counts = {}
# • self.word_counts = {}
# • self.class_word_totals = {}
# • self.vocab = set()

# Phase 2: Training (classifier.train(train_docs, train_labels))
# The model loops through your 12 training documents. Let's look at what the variables contain once training finishes:
# 1. self.class_counts
# Counts how many documents belong to each class.
# • {'spam': 5, 'ham': 7} (Total = 12 documents)

# 2. self.vocab
# Gathers every single unique lowercase word across all 12 documents.
# • {'win', 'free', 'money', 'now', 'lottery', 'ticket', 'winner', 'claim', 'your', 'prize', 'today', 'urgent', 'offer', 'cash', 'congratulations', 'you', 'won', 'meeting', 'tomorrow', 'at', 'noon', 'project', 'update', 'attached', 'can', 'we', 'schedule', 'a', 'call', 'quarterly', 'report', 'review', 'lunch', 'on', 'thursday', 'sounds', 'good', 'team', 'standup', 'notes', 'please', 'the', 'pull', 'request'}
# • vocab_size (len) = 44 unique words.

# 3. self.class_word_totals
# Counts the absolute total number of words processed in each category.
# • spam: 21 total words (e.g., "win free money now" has 4, "free lottery ticket winner" has 4...)
# • ham: 32 total words.

# 4. self.word_counts
# Tracks how many times a word appeared in a specific class. Notice how the word "free" dominates the spam side:
# • 'spam': {'free': 5, 'money': 1, 'win': 1, 'now': 1, 'lottery': 1, 'ticket': 1, 'winner': 1, 'claim': 1, 'your': 1, 'prize': 1, 'today': 1, 'urgent': 1, 'offer': 1, 'cash': 1, 'congratulations': 1, 'you': 1, 'won': 1}
# • 'ham': {'attached': 2, 'review': 2, 'meeting': 1, 'tomorrow': 1, 'at': 1, 'noon': 1, 'project': 1, 'update': 1, 'can': 1, 'we': 1, 'schedule': 1, 'a': 1, 'call': 1, 'quarterly': 1, 'report': 1, 'lunch': 1, 'on': 1, 'thursday': 1, 'sounds': 1, 'good': 1, 'team': 1, 'standup': 1, 'notes': 1, 'please': 1, 'the': 1, 'pull': 1, 'request': 1}

# Phase 3: Prediction (classifier.predict(msg))
# Let’s trace exactly what happens when the loop processes the test messages:
# Example 1: "free money waiting for you"
# The text is split into: ['free', 'money', 'waiting', 'for', 'you'].
# The Model evaluates the spam class:
# 1. Prior Score: Starts at log(5 / 12) = -0.875
# 2. Word Loop:
# 	• "free": Appeared 5 times in spam. Math: log((5+1) / (21 + 1 * 44)) = log(6/65) = -2.383
# 	• "money": Appeared 1 time in spam. Math: log((1+1) / 65) = log(2/65) = -3.481
# 	• "waiting": Unseen word! (Count = 0). Math: log((0+1) / 65) = log(1/65) = -4.174
# 	• "for": Unseen word! (Count = 0). Math: log((0+1) / 65) = -4.174
# 	• "you": Appeared 1 time in spam. Math: log((1+1) / 65) = -3.481
# 3. Total Spam Score: -0.875 + (-2.383) + (-3.481) + (-4.174) + (-4.174) + (-3.481) = -18.568
# The Model evaluates the ham class:
# 1. Prior Score: Starts at log(7 / 12) = -0.539
# 2. Word Loop:
# 	• Note that the denominator for ham is (32 + 1 * 44) = 76.
# 	• "free" (0 times in ham): log(1 / 76) = -4.331
# 	• "money" (0 times in ham): log(1 / 76) = -4.331
# 	• "waiting" (0 times in ham): log(1 / 76) = -4.331
# 	• "for" (0 times in ham): log(1 / 76) = -4.331
# 	• "you" (0 times in ham): log(1 / 76) = -4.331
# 3. Total Ham Score: -0.539 + 5 * (-4.331) = -22.194
# Decision: Because -18.568 is greater than -22.194 (closer to zero), spam wins.

# Final Code Execution Output
# If you run your code snippet, this is exactly what prints to your console:
# text
#   'free money waiting for you' -> spam
#   'meeting rescheduled to friday' -> ham
#   'you won a free prize' -> spam
#   'please review the attached report' -> ham
# Use code with caution.
# • Why did they win?
# 	• "meeting rescheduled to friday" contains "meeting", which only ever appeared in ham during training, driving the ham score up.
# 	• "you won a free prize" contains "you", "won", "free", and "prize", which all appeared in spam, making spam the clear mathematical winner.
# 	• "please review the attached report" contains multiple strong ham indicator words like "please", "review", "attached", and "report".
