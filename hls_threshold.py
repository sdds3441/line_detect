import numpy as np
import cv2
import pickle
img= cv2.imread('C:\\Users\\201921343\\Desktop\\pictures\\straight_lines1.jpg')
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
l_channel = hls[:, :, 1]
s_channel = hls[:, :, 2]
h_channel = hls[:, :, 0]
    # Sobel x
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
cv2.imshow("",h_channel)
cv2.waitKey(0)