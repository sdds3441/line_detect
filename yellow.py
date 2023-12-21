import numpy as np
import cv2

color = [255, 255, 255] #BGR색상에서 녹색
pixel = np.uint8([[color]]) #색상 정보를 하나의 픽셀로 변환한다

hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)

hsv = hsv[0][0] #색상 정보만 가져온다

print("bgr: ", color)
print("hsv: ", hsv)