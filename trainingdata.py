import cv2
import numpy as np
import os.path as path
svm = cv2.ml.SVM_create()

samples, labels = [], []

traindata = cv2.ml.TrainData_create(np.float32(samples), cv2.ml.ROW_SAMPLE, np.float32(labels))

def imgpath(img,classificador):
  return path.join("BrainData", "TrainImages", "%s-%d.jpg" %classificador %img
  
positives = [x for x in range (1,100)]

imgnames = [imgpath(x) for x in positives]

images = [cv2.imread(imgpath(x)) for x in positives]

pos_images = [np.array(np.ravel(cv2.imread(imgpath(x,'pos'))), dtype=np.float32)
              for x in range(1,100)]



