import cv2
from matplotlib import pyplot as plt
import numpy as np
src = cv2.imread("C:\\Users\\201921343\\Desktop\\bus.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#cv2.thereshold(회색이미지 변수명, 임계값, 최댓값, 이진화 종류)
hist = cv2.calcHist(gray, [0], None, [256], [0, 256])
cv2.imshow("dst", dst)
plt.plot(hist,)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()