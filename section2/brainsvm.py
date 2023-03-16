import cv2
import numpy as np
import os.path as path
from lib.utils import imgpath


detect = cv2.xfeatures2d.SIFT_create(15)
extract = cv2.xfeatures2d.SIFT_create()

def extract_feat(img):
    return np.array(np.ravel(extract.compute(img, detect.detect(img))[1]), dtype=np.float32)

traindir = path.join("..","BrainData","TestImages")
images = [cv2.imread(imgpath(traindir, "test", i)) for i in range(1, 20)]

print (len(images))

kps = [extract_feat(img) for img in images]

kps = [kp for kp in kps if len(kp) == 15 * 128]
print ("keypoints: %d" % len(kps[0]))

def test(svm):
    total = 0
    counter = 0
    for k in kps:
        p = svm.predict(np.array([k]))[1][0][0]
        if (p == 1.):
            total += 1.
        counter += 1.

    print ("Acurácia: %f" % (total/counter * 100.0))


