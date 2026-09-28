import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading DNA dataset...")

# Directly load dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"

data = pd.read_csv(
    url,
    names=["Class", "ID", "Sequence"]
)

# Remove unnecessary ID
data = data.drop(columns=["ID"])

# Clean DNA sequences
data["Sequence"] = data["Sequence"].str.replace("\t", "", regex=False)
data["Sequence"] = data["Sequence"].str.lower()

sequences = data["Sequence"].tolist()
labels = data["Class"].tolist()

print("Dataset loaded successfully!")
print("Number of sequences:", len(sequences))

# Convert DNA sequences into numerical features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 3)
)

X = vectorizer.fit_transform(sequences)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Create AI model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n--- AI MODEL RESULTS ---")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions))

# New DNA prediction
print("\n--- DNA PROMOTER PREDICTION ---")

new_sequence = input("Enter a DNA sequence: ")
new_sequence = new_sequence.strip().lower()

if any(base not in "atgc" for base in new_sequence):
    print("Invalid DNA sequence. Use only A, T, G and C.")

else:
    new_features = vectorizer.transform([new_sequence])
    prediction = model.predict(new_features)[0]

    if prediction == "+":
        print("\nPrediction: PROMOTER SEQUENCE")
    else:
        print("\nPrediction: NON-PROMOTER SEQUENCE")

print("\nThis is an educational machine-learning prediction.")