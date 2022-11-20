
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
import argparse
import joblib
from sklearn.metrics import accuracy_score as metric
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--clf_name', type = str, required = True, help='enter classifier name')
parser.add_argument('--random_state', type = int, required = True, help='enter classifier name')
args = parser.parse_args()

digits = datasets.load_digits()


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf_svm = svm.SVC(gamma=0.001)
clf_dect = tree.DecisionTreeClassifier()

def get_metrics(y_test, predicted):
    accuracy = metric(y_test, predicted)
    f1_score_ = f1_score(y_test, predicted, average='macro')
    return accuracy, f1_score_


def split(random_state):
# Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=True, random_state = random_state
    )
    return X_train, X_test, y_train, y_test

def train_classifier(classifier, random_state):

    X_train, X_test, y_train, y_test = split(random_state)

    if classifier == 'svm':
        clf_svm.fit(X_train, y_train)
        print('SVM Model Trained with random state {}'.format(random_state))
        path = joblib.dump(clf_svm, 'models/{}_{}.pkl'.format(classifier, random_state))
        predict = clf_svm.predict(X_test)
        acccuracy, f1_score = get_metrics(y_test, predict)
        with open('results/{}_{}.txt'.format(classifier, random_state), 'w+') as file:
            file.write('test accuracy: {}\n'.format(acccuracy))
            file.write('test macro-f1: {}\n'.format(f1_score))
            file.write('model saved at {}'.format(path))

    else:
        clf_dect.fit(X_train, y_train)
        print('Decision Tree Model Trained with random state {}'.format(random_state))
        path = joblib.dump(clf_dect, 'models/{}_{}.pkl'.format(classifier, random_state))
        predict = clf_dect.predict(X_test)
        acccuracy, f1_score = get_metrics(y_test, predict)
        with open('results/{}_{}.txt'.format(classifier, random_state), 'w+') as file:
            file.write('test accuracy: {}\n'.format(acccuracy))
            file.write('test macro-f1: {}\n'.format(f1_score))
            file.write('model saved at {}'.format(path))


train_classifier(args.clf_name, args.random_state)













