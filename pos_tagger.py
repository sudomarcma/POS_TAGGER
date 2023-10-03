import gzip
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

def read_data(filename):
    with gzip.open(filename, 'rt') as f:
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

def train_model(model_choice):
    train_data = read_data('train.txt.gz')
    X_train = [[features(sentence, index) for index in range(len(sentence))] for sentence in [list(zip(*sentence))[0] for sentence in train_data]]
    y_train = [list(zip(*sentence))[1] for sentence in train_data]

    # Flatten the training data and labels
    X_train = [item for sublist in X_train for item in sublist]
    y_train = [item for sublist in y_train for item in sublist]

    if model_choice == 1:
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', MultinomialNB())
        ])
        model_filename = 'bayesian_classifier_model.pkl'
    elif model_choice == 2:
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1))
        ])
        model_filename = 'logistic_regression_model.pkl'
    elif model_choice == 3:
        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', LinearSVC())
        ])
        model_filename = 'svm_model.pkl'

    clf.fit(X_train, y_train)

    # Save the trained model
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)

    print(f"{model_filename} trained and saved!")

if __name__ == "__main__":
    print("Select a model to train:")
    print("1: Bayesian Classifier")
    print("2: Logistic Regression")
    print("3: SVM")
    choice = int(input("Enter your choice (1/2/3): "))

    train_model(choice)