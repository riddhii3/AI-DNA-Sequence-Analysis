# AI-Based DNA Promoter Sequence Classification and Analysis

## Project Overview

This project uses Artificial Intelligence, Machine Learning, and Bioinformatics techniques to analyse DNA sequences and classify them as promoter or non-promoter sequences.

The system performs basic DNA sequence analysis and uses a Machine Learning model to predict whether a given DNA sequence is a promoter sequence.

## Objectives

- Analyse DNA sequences.
- Calculate DNA sequence length.
- Count A, T, G and C nucleotides.
- Calculate GC percentage.
- Classify DNA sequences as promoter or non-promoter.
- Apply Machine Learning techniques to biological sequence data.

## Features

### DNA Sequence Analysis
The program calculates:

- DNA sequence length
- Number of Adenine (A)
- Number of Thymine (T)
- Number of Guanine (G)
- Number of Cytosine (C)
- GC percentage

### AI-Based Classification

The project uses:

- Character-level TF-IDF for feature extraction
- Logistic Regression for classification

The trained model predicts whether the input DNA sequence is:

- PROMOTER SEQUENCE
- NON-PROMOTER SEQUENCE

## Dataset

The project uses the **UCI Molecular Biology Promoter Gene Sequences Dataset**.

The dataset contains labelled DNA sequences from *E. coli* that are used to train and test the Machine Learning model.

Dataset Source:

https://archive.ics.uci.edu/dataset/67/molecular%2Bbiology%2Bpromoter%2Bgene%2Bsequences

## Methodology

The project follows these steps:

1. Load the DNA promoter dataset.
2. Clean and preprocess the DNA sequences.
3. Convert DNA sequences into numerical features using character-level TF-IDF.
4. Split the dataset into training and testing data.
5. Train a Logistic Regression model.
6. Evaluate the model using test accuracy.
7. Accept a new DNA sequence from the user.
8. Calculate DNA sequence statistics.
9. Predict whether the sequence is a promoter or non-promoter.

## Workflow

```text
DNA Sequence
     ↓
Data Cleaning
     ↓
TF-IDF Feature Extraction
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
New DNA Sequence
     ↓
Promoter / Non-Promoter Prediction
