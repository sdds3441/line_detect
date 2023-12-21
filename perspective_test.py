import numpy as np
import cv2
import time
#src = np.array([(305, 198),(626, 198),(105,305),(738, 306)], dtype=np.float32) #right
src=np.array([[408, 186], [725, 184], [305, 291], [936, 295]], dtype=np.float32) #left
dst=np.array([[365, 400], [635, 400], [365, 670], [635, 670]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src, dst)
M2 = cv2.getPerspectiveTransform(dst, src)
lane5=[]
shape=[0,0,0]
def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 이미지를 1:1비율로 OverLap하는 함수
   return cv2.addWeighted(initial_img, α, img, β, λ)

def perspective_warp(img, dst_size=(1000, 1000)):  # 이미지를 버드뷰로 변환
   #cv2.circle(img, (src[0][0], src[0][1]), 4, (0, 255, 0), -1)  # 좌상
   #cv2.circle(img, (src[1][0], src[1][1]), 4, (0, 255, 255), -1)  # 우상
   #cv2.circle(img, (src[2][0], src[2][1]), 4, (255, 255, 0), -1)  # 우하
   #cv2.circle(img, (src[3][0], src[3][1]), 4, (255, 0, 255), -1)  # 좌하
   warped = cv2.warpPerspective(img, M, dst_size)
   return warped
img = cv2.imread("C:/Users/201921343/Desktop/testl.png")
savs=np.dot(M,[[288], [500], [1]]).T[0]
lane5.append(savs[:2]/savs[2])
warp=perspective_warp(img)
cv2.circle(warp, (int(lane5[0][0]), int(lane5[0][1])), 10, (255, 0, 0), -1)  # 좌하
cv2.imshow('testo', warp)
shape[0],shape[1],shape[2]=warp.shape
dist=int(lane5[0][1])
print(shape[1]-dist)
cv2.waitKey(0)