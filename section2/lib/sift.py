import cv2
import numpy as np
import os.path as path

detector = cv2.xfeatures2d.SIFT_create(15)
extractor = cv2.xfeatures2d.SIFT_create()
traindir = path.join("..","BrainData","TrainImages")
def extract_features(klass, image):
    img = cv2.imread(imgpath(traindir, klass, image))
    return np.array(np.ravel(extract.compute(img, detect.detect(img))[1])[0 : LEN_FEATURES], dtype=np.float32)
