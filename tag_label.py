import pickle

def read_data(filename):
    with open(filename, 'rt') as f:
        content = f.read().strip().split('\n\n')
        sentences = [[tuple(row.split()[:2]) for row in sentence.split('\n')] for sentence in content]
    return sentences

def features(sentence, index):
    #Compute features for a given sentence and token index
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    }

def tag_sentences(model_filename, test_filename, output_filename):
    # Load the selected model
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)

    # Read the unlabeled test data
    test_data = read_data(test_filename)
    X_test = [[features(sentence, index) for index in range(len(sentence))] for sentence in [list(zip(*sentence))[0] for sentence in test_data]]

    # Predict the POS tags using the loaded model
    y_pred = [clf.predict(sentence_features) for sentence_features in X_test]

    # Write the tokens and their predicted POS tags to the output file
    with open(output_filename, 'w') as f:
        for tokens, tags in zip(test_data, y_pred):
            for token, tag in zip(tokens, tags):
                f.write(f"{token[0]}\t{tag}\n")
            f.write("\n")

if __name__ == "__main__":
    print("Select a model for POS tagging:")
    print("1: Bayesian Classifier")
    print("2: Logistic Regression")
    print("3: SVM")
    choice = int(input("Enter your choice (1/2/3): "))

    model_files = {
        1: "bayesian_classifier_model.pkl",
        2: "logistic_regression_model.pkl",
        3: "svm_model.pkl"
    }
    output_files = {
        1: "Bayesian_Classifier_pos.txt",
        2: "Logistic_Regression_pos.txt",
        3: "SVM_pos.txt"
    }

    model_filename = model_files[choice]
    output_filename = output_files[choice]
    test_filename = 'unlabeled_test_test.txt'

    tag_sentences(model_filename, test_filename, output_filename)
    print(f"POS tagging completed! Results saved to {output_filename}")
