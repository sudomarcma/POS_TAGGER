# POS_TAGGER
Bayesian Classifier, Logistic Regression, and SVM for Part of Speech Tagging. Penn State CMPSC 448 FA23

This POS Tagger is a Python-based tool that uses machine learning models to predict the Part-of-Speech tags for given sentences.

## Features

- Train individual models on the provided dataset.
- Predict POS tags for given sentences using trained models.
- Evaluate the accuracy of each model.
- Token-wise comparison of predicted tags with the gold standard.
- Sentence-wise comparison of predicted tags with the gold standard.
- Generate an evaluation report detailing the performance and differences between models.

## Requirements

- Python 3.7+
- Scikit-learn
- Gzip
- Pickle

## Usage

1. **Changing the Corpus and Gold Standard**:  
- In `pos_tagger.py` replace `'train.txt.gz'` with the path to your desired corpus file.
- In `evaluate.py` replace `'train.txt.gz'` with the path to your Gold Standard file.

2. **Training Models**:  
Run the `pos_tagger.py` script and select the desired model to train.

3. **Tag Models**  
Run the `tag_label.py` script and select the desired model to tag.

4. **Evaluating Models**  
Run the `evaluate.py` script to generate an evaluation report.
