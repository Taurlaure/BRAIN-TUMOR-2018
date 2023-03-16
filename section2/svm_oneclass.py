import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from lib.utils import imgpath
import brainsvm



traindir = path.join("..","BrainData","TrainImages")
print (path.realpath(traindir))

POS = 200
NEG = 100
NUM_FEATURES = 15
LEN_FEATURES = NUM_FEATURES * 128

detect = cv2.xfeatures2d.SIFT_create(15)
extract = cv2.xfeatures2d.SIFT_create()

# def imgpath(basepath, klass, number):
#     return path.join(basepath, "%s-%d.pgm" % (klass, number))

def extract_features(klass, image):
    img = cv2.imread(imgpath(traindir, klass, image))
    return np.array(np.ravel(extract.compute(img, detect.detect(img))[1])[0 : LEN_FEATURES], dtype=np.float32)

positives = [extract_features("pos", img) for img in range(1, POS + 1)]
negatives = [extract_features("neg", img) for img in range(1, NEG + 1)]

positives = [p for p in positives if len(p) == LEN_FEATURES]
negatives = [n for n in negatives if len(n) == LEN_FEATURES]




trainimages = positives + negatives
trainlabels = np.append(np.ones(len(positives)), np.zeros(len(negatives))) 
print ("Img: %d, responses: %d" % (len(trainimages), len(trainlabels)))
traindata = cv2.ml.TrainData_create(np.array(trainimages), cv2.ml.ROW_SAMPLE, np.array(trainlabels ,dtype=np.int32))

np.savetxt('x.txt',positives)
np.savetxt('labels.txt',negatives)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(1)
svm.setC(1)
svm.setTermCriteria(criteria)

svm.train(traindata)

svm.save("brainsvm.data")
brainsvm.test(svm)

