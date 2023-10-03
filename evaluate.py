import gzip
import pickle

# Function to read the tagged data
def read_tagged_data(filename, gzipped=False):
    open_func = gzip.open if gzipped else open
    mode = 'rt'
    
    with open_func(filename, mode) as f:
        content = f.read().strip().split('\n\n')
        # Only consider the token and POS tag, ignore the chunking tag
        sentences = [[tuple(row.split()[:2]) for row in sentence.split('\n')] for sentence in content]
    return sentences

def compute_token_accuracy(predicted_file, gold_standard_file):
    predicted_data = read_tagged_data(predicted_file)
    gold_standard_data = read_tagged_data(gold_standard_file, gzipped=True)

    correct_tags = 0
    total_tags = 0

    # Flatten the data to get a list of tokens and their tags
    predicted_tokens = [token for sentence in predicted_data for token in sentence]
    gold_standard_tokens = [token for sentence in gold_standard_data for token in sentence]

    # Iterate over tokens in the predicted data
    for token in predicted_tokens:
        if token in gold_standard_tokens:
            correct_tags += 1
        total_tags += 1

    return correct_tags / total_tags if total_tags != 0 else 0

gold_standard_file = 'train.txt.gz'  # Your gzipped gold standard file

model_files = {
    "Bayesian Classifier": "Bayesian_Classifier_pos.txt",
    "Logistic Regression": "Logistic_Regression_pos.txt",
    "SVM": "SVM_pos.txt"
}

# Token-wise accuracy comparison
results = ["Token-wise Accuracy Comparison:\n"]
accuracies = {}
for model_name, output_file in model_files.items():
    accuracy = compute_token_accuracy(output_file, gold_standard_file)
    accuracies[model_name] = accuracy
    results.append(f"{model_name} Token-wise Accuracy: {accuracy:.4f}\n")

most_accurate_model = max(accuracies, key=accuracies.get)
results.append(f"\nThe most accurate model token-wise is: {most_accurate_model} with an accuracy of {accuracies[most_accurate_model]:.4f}\n")

# Model accuracy comparison
results.append("\nModel Accuracy Comparison:\n")

# Feature extraction function
def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    }

# Compare tagged files and return differences
def compare_tagged_files(file1, file2, file3):
    data1 = read_tagged_data(file1)
    data2 = read_tagged_data(file2)
    data3 = read_tagged_data(file3)

    differences = []

    for sent1, sent2, sent3 in zip(data1, data2, data3):
        for (word1, tag1), (word2, tag2), (word3, tag3) in zip(sent1, sent2, sent3):
            if word1 == word2 == word3 and (tag1 != tag2 or tag1 != tag3 or tag2 != tag3):
                differences.append((word1, tag1, tag2, tag3))

    return differences

# Load the models
with open('bayesian_classifier_model.pkl', 'rb') as f:
    bayesian_model = pickle.load(f)
with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Read the gold standard data
gold_standard_data = read_tagged_data('train.txt.gz', gzipped=True)
X_gold = [[features(sentence, index) for index in range(len(sentence))] for sentence in gold_standard_data]
y_gold = [[tag for word, tag in sentence] for sentence in gold_standard_data]

# Predict and compute accuracy for each model
models = {
    'Bayesian Classifier': bayesian_model,
    'Logistic Regression': logistic_regression_model,
    'SVM': svm_model
}

# Save initial results to evaluate.txt
with open("evaluate.txt", "w") as f:
    f.writelines(results)

# Append model accuracy and differences to evaluate.txt
with open("evaluate.txt", "a") as eval_file:
    for name, model in models.items():
        correct_tags = 0
        total_tags = 0
        for sentence_features, gold_tags in zip(X_gold, y_gold):
            predicted_tags = model.predict(sentence_features)
            correct_tags += sum(p == g for p, g in zip(predicted_tags, gold_tags))
            total_tags += len(gold_tags)
        accuracy = correct_tags / total_tags
        eval_file.write(f"{name} Model Accuracy: {accuracy:.4f}\n")

    # Compare tagged files and write differences to evaluate.txt
    differences = compare_tagged_files("Bayesian_Classifier_pos.txt", "Logistic_Regression_pos.txt", "SVM_pos.txt")
    eval_file.write("\nToken differences between the files:\n")
    eval_file.write("-" * 80 + "\n")
    eval_file.write(f"{'Word':<20}{'Bayesian':<20}{'Logistic Regression':<20}{'SVM':<20}\n")
    eval_file.write("-" * 80 + "\n")
    for word, tag1, tag2, tag3 in differences:
        eval_file.write(f"{word:<20}{tag1:<20}{tag2:<20}{tag3:<20}\n")
    eval_file.write("-" * 80 + "\n")

print("Evaluation results saved to evaluate.txt")
