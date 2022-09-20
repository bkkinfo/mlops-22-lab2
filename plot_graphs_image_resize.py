import matplotlib.pyplot as plt
     
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
    
    
gamma_list = [0.01 ,0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1 ,0.2, 0.5, 0.7 ,1 ,2, 5 ,7,10]
    
h_params_comb = [{'gamma':g,"C":c} for g in gamma_list for c in c_list]
    
assert len(h_params_comb) == len(gamma_list)*len(c_list)
    
#model hyperparameter
    
    
    
    
    
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1
#PART: load datsets -data from csv,tsv,json,pickel
import matplotlib.pyplot as plt
    
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import transform
digits = datasets.load_digits()
features=digits.data
targets=digits.target
    
    
resoArr = [4, 16, 256]
resimg = []
for res in resoArr:
#     for i in range(len(features)):
#         print (digits.data.shape)
        # newfeatures2=transform.resize(features[i].reshape(8,8),(res,res))
    
    newfeatures2=[transform.resize(features[i].reshape(8,8),(res,res))for i in range(len(features))]
    for k in range (10):
        resimg.append (newfeatures2[k].reshape(res,res))
    
# plt.imshow(resimg[0])
# plt.show()
    
    
    
# newfeatures2=[transform.resize(features[i].reshape(8,8),(256,256))for i in range(len(features))]
# for i in range(4):
#   x = newfeatures2[i].reshape((128,128))
#   plt.imshow(x)
fig = plt.figure(figsize=(10,5))
imgArr=[]
n= len(resimg)
for i in range(n):
    fig.add_subplot(3,n//3,i+1)
    #   x = newfeatures2[i].reshape((8,8))
    x = resimg[i]
    imgArr.append(x[:,:])
    
    plt.imshow(x)
    if i == 5:
        plt.title("Resolution: "+str(resoArr[0]))
    elif i == 15:
        plt.title("Resolution: "+str(resoArr[1]))
    elif i == 25:
        plt.title("Resolution: "+str(resoArr[2]))
    
    
    plt.axis('off')
    
#PART: sanity check visyualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    
###############################################################################
    
#PART: data prprocessing  -- to remove some noise, to normalize data,format the data to be consumed by nod
# flatten the images
n_samples = len(digits.images)
    
############################################
print("\nSize of Images in digits dataset\t" + str(digits.images.shape)+"\n")

#############################################
data = digits.images.reshape((n_samples, -1))
    
#PART: define train/dev/test splits of experiments protocol
# Split data into 50% train and 50% test subsets
    
#80:10:10 train:dev:test
#define model
dev_test_frac = 1-train_frac
X_train,X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
    
    
X_test,X_dev, y_test, y_dev = train_test_split(
    X_dev_test,y_dev_test, test_size=(dev_frac)/(dev_test_frac), shuffle=True
)
    
#if tsting on the same as training set: the performance metrics may overstimate the goodness of the model
#you want to test on "unseen" samples.
#train to train model
#dev to set hyperparameters of the model
#test to evaluate the performance of the model
    
#part : definf 
# Create a classifier: a support vector classifier
best_acc = -1.0
best_model = None
best_h_params = None
for cur_h_params in h_params_comb:
    GAMMA = 0.001
    C = 0.5
    clf = svm.SVC()
    
    #part: setting up hyperparameter
    hyper_params = {'gamma':GAMMA,"C":C}
    clf.set_params(**hyper_params)
    
    
    
    
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    #PART: 
    # Predict the value of the digit on the test subset
    predicted_dev = clf.predict(X_dev)
    cur_acc = metrics.accuracy_score(y_pred = predicted_dev,y_true = y_dev)
    
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_model = clf
        best_h_params = cur_h_params
        print("found new best acc with: "+str(cur_h_params))
        print("New best val accuracy:"+str(cur_acc))
    
predicted = best_model.predict(X_test)
    
#PART: get test set predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    
predicted_train = clf.predict(X_train)
predicted_dev = clf.predict(X_dev)
predicted_test = clf.predict(X_test)

accuracy_train = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
accuracy_dev = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
accuracy_test = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)   
    
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
    
# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
plt.show()
    
print("Best hyperparameters were: ")
print(cur_h_params,"Train ACC", round(accuracy_train, 3), "\t","Train DEV", round(accuracy_dev, 3), "\t","Train TEST", round(accuracy_test, 3),"Best Combination\n")
