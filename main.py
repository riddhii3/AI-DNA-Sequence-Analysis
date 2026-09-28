# AI-Based DNA Promoter Sequence Classification and Analysis

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("==========================================")
print(" AI-BASED DNA SEQUENCE ANALYSIS SYSTEM")
print("==========================================")

# -------------------------------
# STEP 1: Load DNA Dataset
# -------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"

import pandas as pd

data = pd.read_csv(
    url,
    names=["Class", "ID", "Sequence"]
)

data = data.drop(columns=["ID"])

data["Sequence"] = data["Sequence"].str.replace("\t", "", regex=False)
data["Sequence"] = data["Sequence"].str.lower()

sequences = data["Sequence"].tolist()
labels = data["Class"].tolist()

print("\nDataset loaded successfully!")
print("Total DNA sequences:", len(sequences))


# -------------------------------
# STEP 2: Train AI Model
# -------------------------------

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 3)
)

X = vectorizer.fit_transform(sequences)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("AI Model Accuracy:", round(accuracy * 100, 2), "%")


# -------------------------------
# STEP 3: User DNA Sequence
# -------------------------------

print("\n------------------------------------------")
print(" DNA SEQUENCE ANALYSIS")
print("------------------------------------------")

dna = input("Enter a DNA sequence: ")

dna = dna.strip().lower()

# Check valid DNA
if any(base not in "atgc" for base in dna):

    print("\nInvalid DNA sequence!")
    print("Please use only A, T, G and C.")

else:

    # -------------------------------
    # STEP 4: DNA Analysis
    # -------------------------------

    length = len(dna)

    A = dna.count("a")
    T = dna.count("t")
    G = dna.count("g")
    C = dna.count("c")

    if length > 0:
        gc_percentage = ((G + C) / length) * 100
    else:
        gc_percentage = 0

    print("\n========== DNA ANALYSIS ==========")

    print("DNA Sequence:", dna.upper())
    print("DNA Length:", length)
    print("A Count:", A)
    print("T Count:", T)
    print("G Count:", G)
    print("C Count:", C)
    print("GC Percentage:", round(gc_percentage, 2), "%")


    # -------------------------------
    # STEP 5: AI Prediction
    # -------------------------------

    new_features = vectorizer.transform([dna])

    prediction = model.predict(new_features)[0]

    print("\n========== AI PREDICTION ==========")

    if prediction == "+":
        print("Prediction: PROMOTER SEQUENCE")
    else:
        print("Prediction: NON-PROMOTER SEQUENCE")


    print("\n==========================================")
    print(" Analysis Completed Successfully!")
    print("==========================================")