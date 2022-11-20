import sys, os
import numpy as np
from joblib import load

sys.path.append(".")

from utils import get_hyperparameters, tune_and_save
from sklearn import svm, metrics

def test_get_hyperparameters():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_hyperparameters("svc", params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_hyperparameters("svc", params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = (metrics.accuracy_score, metrics.f1_score)
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(h_param_comb, clf, x_train, y_train, x_dev, y_dev, x_dev, y_dev, metric, 51, "./", "./", model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)