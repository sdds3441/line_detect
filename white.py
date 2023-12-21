 #-*- coding: utf-8 -*-  # 한글 주석쓰려면 이거 해야함
import cv2 # opencv 사용
import numpy as np
import time
start = time.time()
image = cv2.imread('C:\\Users\\201921343\\Desktop\\pictures\\solidWhiteCurve.jpg') # 이미지 읽기
mark = np.copy(image) # image 복사

#  BGR 제한 값 설정
blue_threshold = 200
green_threshold = 200
red_threshold = 200
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로
thresholds = (image[:,:,0] < bgr_threshold[0]) \
            | (image[:,:,1] < bgr_threshold[1]) \
            | (image[:,:,2] < bgr_threshold[2])
mark[thresholds] = [0,0,0]

cv2.imshow('white',mark) # 흰색 추출 이미지 출력
cv2.imshow('result',image) # 이미지 출력
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
cv2.waitKey(0)