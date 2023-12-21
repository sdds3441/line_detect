import cv2
import time
start = time.time()
src = cv2.imread("C:\\Users\\201921343\\Desktop\\pictures\\solidWhiteCurve.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.inRange(h, 0, 255)
blue = cv2.bitwise_and(hsv, hsv, mask = v)
blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)
dst=cv2.fastNlMeansDenoisingColored(blue,None,60,10,7,21)

cv2.imshow("busline", blue)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
cv2.waitKey(0)

cv2.destroyAllWindows()
