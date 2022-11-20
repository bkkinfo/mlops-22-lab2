import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, metrics
import pandas as pd
import copy
from joblib import dump

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def data_viz(digits):
    _, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 5))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 5))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()

def confusion_matrix_viz(y_test, predictions, clf):
    cm = metrics.confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

def train_dev_test_split(data, label, train_frac, dev_frac, random_state_val):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True, random_state=random_state_val
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True, random_state=random_state_val
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def get_hyperparameters(algorithm, params):
    if algorithm == "svc":
        params_combinations = [{"gamma":g, "C":c} for g in params['gamma'] for c in params['C']]
    elif algorithm == "dtc":
        params_combinations = [{"ccp_alpha":c, "min_impurity_decrease":i} for c in params['ccp_alpha'] for i in params['min_impurity_decrease']]
    else:
        params_combinations = []
    return params_combinations

def hyperparam_search(hyper_params, clf, X_train, y_train, X_test, y_test, X_dev, y_dev, metric):
    best_metric, best_model, best_h_params = -0.1, None, None
    
    for param in hyper_params:
        # print(param)
        clf.set_params(**param)
        clf.fit(X_train, y_train)
        pre_dev = clf.predict(X_dev)
        cur_metric = metric(y_pred=pre_dev, y_true=y_dev)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = copy.copy(clf)
            best_h_params = param
            # print("Found new best with:" + str(param))
            # print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params

def tune_and_save(hyper_params, clf, X_train, y_train, X_test, y_test, X_dev, y_dev, metric, random_state, model_dir, result_dir, model_path):
    best_model, best_metric, best_h_params = hyperparam_search(
        hyper_params,
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        X_dev,
        y_dev,
        metric[0]
    )
    
    pre_dev = clf.predict(X_dev)
    macro_f1_metric = metric[1](y_pred=pre_dev, y_true=y_dev, average='macro')

    best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )
    if type(clf) == svm.SVC:
        model_type = 'svm'
    elif type(clf) == tree.DecisionTreeClassifier:
        model_type = 'tree'
    else:
        model_type = 'none'

    best_model_file = model_dir + model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_file
    dump(best_model, model_path)

    # print(f"Model: {model_type}")
    # print(f"Best hyperparameters: {best_h_params}")
    # print(f"Best metric on Dev data: {best_metric}")
    # print("")

    f = open(f"{result_dir}{model_type}_{random_state}.txt", "w")
    
    f.write(f"test accuracy: {best_metric}\n")
    f.write(f"test macro-f1: {macro_f1_metric}\n")
    f.write(f"model saved at {model_path}")
    f.close()

    return model_path
