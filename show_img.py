import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

def undistort(img, cal_dir='C:\\Users\\201921343\\Desktop\\cal_pickle.p'):
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

img = cv2.imread('C:\\Users\\201921343\\Desktop\\pictures\\trans_dotted.png')
dst = cv2.imread('C:\\Users\\201921343\\Desktop\\pictures\\dotted_original.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Dotted Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Original Image', fontsize=30)
plt.show()
cv2.waitKey(0)