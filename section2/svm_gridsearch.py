
import sklearn.svm as svm
import cv2
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

NUM_CLASSES = 2
NUM_SAMPLES = 300
img_path = "../BrainData/TrainImages/%s-%s.jpg"
target_names = ["pos", "neg"]
X, y = [], []

def rng(x):
    return range(1, x + 1)

def read(cls, id):
    print ("%s,%s" % (cls, id))
    return np.ravel(cv2.imread(img_path % (cls, id), 0))

for x in range(0,NUM_CLASSES):
    for i in rng(NUM_SAMPLES):
        X.append(read(target_names[x], i))
        y.append(x)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)

svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))
