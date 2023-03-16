import cv2
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os.path as path
from plot_cm import plot_confusion_matrix

import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from lib.utils import read_data, plot_data, plot_decision_function

NUM_SAMPLES_POS=301
NUM_SAMPLES_NEG=200
NUM_SAMPLES = 501
POS_CLASS = "pos"
NEG_CLASS = "neg"
NUM_FEATURES = 20
FEAT_LEN = NUM_FEATURES * 128

def get_image(clss, num):
    return cv2.imread(path.join("..","BrainData", "TrainImages", "%s-%d.jpg" % (clss, num)))

def extract(img):
    return np.ravel(cv2.calcHist([img],[1],None,[126],[0,256]))

pos_samples = [extract(get_image(POS_CLASS, x)) for x in range(1, NUM_SAMPLES_POS)]
neg_samples = [extract(get_image(NEG_CLASS, x)) for x in range(1, NUM_SAMPLES_NEG)]

np.savetxt('positives.txt',pos_samples)
np.savetxt('negatives.txt',neg_samples)

#all_samples = np.concatenate((pos_samples, neg_samples), axis = 0)
#all_labels = np.concatenate((np.ones(len(pos_samples), dtype=np.int32), np.zeros(len(neg_samples), dtype=np.int32)), axis=0)
#all_samples = all_samples / 4000.

#X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42)

#print (X_train[0])
#parameters =  {}
#dt = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', random_state=0, max_depth=15, verbose=1, max_features=20, n_estimators=840)
#clf = GridSearchCV(dt, parameters)
#clf.fit(X_train, y_train)

#print (clf.best_params_)
#y_pred = clf.predict(X_test)

#target_names = [POS_CLASS, NEG_CLASS]
#print(classification_report(y_test, y_pred, target_names=target_names))

#Read data
pos_samples, neg_samples = read_data('positives.txt','negatives.txt')
 
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(pos_samples, neg_samples, test_size = 0.2, random_state=42)
 
# Plot traning and test data
plot_data(X_train, y_train, X_test, y_test)

# Create a linear SVM classifier 
# RBF SVM classifier
#clf = svm.SVC(kernel='rbf',C=1,gamma=1)
clf = svm.SVC(C =1, kernel='rbf', gamma=1)
# Train classifier 
clf.fit(X_train, y_train)
 
# Plot decision function on training and test data
plot_decision_function(X_train, y_train, X_test, y_test, clf)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Acurácia: {}%".format(clf.score(X_test, y_test) * 100))


# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
#Train the classifier
clf_grid.fit(X_train, y_train)

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

