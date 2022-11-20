from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def test_random_state_same():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    clf1 = svm.SVC(gamma=0.001)
    clf2 = svm.SVC(gamma=0.001)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False, random_state = 1337)
    clf1.fit(X_train1, y_train1)
    predicted1 = clf1.predict([[ 0,  0,  1, 11, 14, 15,  3,  0,  0,  1, 13, 16, 12, 16,  8,  0,  0,  8, 16,  4,  6, 16,  5,  0,  0,  5, 15, 11, 13, 14,  0,  0,  0,  0,  2, 12, 16, 13,  0,  0,  0,  0,  0, 13, 16, 16,  6,  0,  0,  0,  0, 16, 16, 16, 7,  0,  0,  0,  0, 11, 13, 12,  1,  0]])
    

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False, random_state = 1337)
    clf2.fit(X_train2, y_train2)
    predicted2 = clf2.predict([[ 0,  0,  1, 11, 14, 15,  3,  0,  0,  1, 13, 16, 12, 16,  8,  0,  0,  8, 16,  4,  6, 16,  5,  0,  0,  5, 15, 11, 13, 14,  0,  0,  0,  0,  2, 12, 16, 13,  0,  0,  0,  0,  0, 13, 16, 16,  6,  0,  0,  0,  0, 16, 16, 16, 7,  0,  0,  0,  0, 11, 13, 12,  1,  0]])

    assert predicted1.all() == predicted2.all()


def test_random_state_different():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    clf1 = svm.SVC(gamma=0.001)
    clf2 = svm.SVC(gamma=0.001)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True, random_state = 1246487484)
    clf1.fit(X_train1, y_train1)
    predicted1 = clf1.predict([[ 0,  0,  1, 11, 14, 15,  3,  0,  0,  1, 13, 16, 12, 16,  8,  0,  0,  8, 16,  4,  6, 16,  5,  0,  0,  5, 15, 11, 13, 14,  0,  0,  0,  0,  2, 12, 16, 13,  0,  0,  0,  0,  0, 13, 16, 16,  6,  0,  0,  0,  0, 16, 16, 16, 7,  0,  0,  0,  0, 11, 13, 12,  1,  0]])
    

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True,random_state = 42)
    clf2.fit(X_train2, y_train2)

    assert X_train1.all() == X_train2.all()
    
    #

