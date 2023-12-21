import cv2
import numpy as np
import time
import math

cap = cv2.VideoCapture('E:/2020.09.12/RIGHTdst1.avi')  # 영상 넣어주세요
frame_cut = 5  # 가져올 프레임 수 if 5, 1번 6번 11번째 이미지를 가져온다.
count = 1  # 데이터 시작 숫자

countf = 0

while (cap.isOpened()):
    start = time.time()

    ret, frame = cap.read()
    if countf % frame_cut == 0:
        path_X ="20%05d" % (count) + ".png"
        frame = cv2.resize(frame, (1024, 576))
        cv2.imwrite('E:/20/' + path_X, frame*0.9)
        count = count + 1
        print(count)
    countf = countf + 1
#dkfldskfs
