import sys, os
import numpy as np
from joblib import load
from sklearn import datasets

sys.path.append(".")

from utils import preprocess_digits, train_dev_test_split

def test_same_seed_same_data():
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    del digits
    train_frac = 0.8
    dev_frac = 0.1
    data_split_random_state_1 = 53
    data_split_random_state_2 = 53
    X_train_1, y_train_1, X_dev_1, y_dev_1, X_test_1, y_test_1 = train_dev_test_split(
        data, label, train_frac, dev_frac, data_split_random_state_1
    )
    X_train_2, y_train_2, X_dev_2, y_dev_2, X_test_2, y_test_2 = train_dev_test_split(
        data, label, train_frac, dev_frac, data_split_random_state_2
    )
    
    assert data_split_random_state_1 == data_split_random_state_2
    assert (X_train_1==X_train_2).all() == True
    assert (X_test_1==X_test_2).all() == True
    assert (X_dev_1==X_dev_2).all() == True
    assert (y_train_1==y_train_2).all() == True
    assert (y_test_1==y_test_2).all() == True
    assert (y_dev_1==y_dev_2).all() == True

def test_different_seed_different_data():
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    del digits
    train_frac = 0.8
    dev_frac = 0.1
    data_split_random_state_1 = 53
    data_split_random_state_2 = 29
    X_train_1, y_train_1, X_dev_1, y_dev_1, X_test_1, y_test_1 = train_dev_test_split(
        data, label, train_frac, dev_frac, data_split_random_state_1
    )
    X_train_2, y_train_2, X_dev_2, y_dev_2, X_test_2, y_test_2 = train_dev_test_split(
        data, label, train_frac, dev_frac, data_split_random_state_2
    )
    
    assert data_split_random_state_1 != data_split_random_state_2
    assert (X_train_1==X_train_2).all() == False
    assert (X_test_1==X_test_2).all() == False
    assert (X_dev_1==X_dev_2).all() == False
    assert (y_train_1==y_train_2).all() == False
    assert (y_test_1==y_test_2).all() == False
    assert (y_dev_1==y_dev_2).all() == False