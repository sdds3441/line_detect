import numpy as np
import pandas as pd
import numpy as np
import cv2
import glob

# copy parameters to arrays
# A (Intrinsic Parameters) [fc, skew*fx, cx], [0, fy, cy], [0, 0, 1]
K = np.array([[760.3371,        0.,           629.0291],
              [0,             763.7501,       378.7569],
              [0,                 0,                 1]])

# Distortion Coefficients(kc) - 1st, 2nd
d = np.array([0.0110684, -0.0085019, 0, 0, 0]) # just use first two terms
capr = cv2.VideoCapture(1)

widthr = int(capr.get(cv2.CAP_PROP_FRAME_WIDTH))
heightr = int(capr.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):

    # 이미지를 캡쳐
    ret, framer = capr.read()

    # 캡쳐되지 않은 경우 처리
    if ret == False:
        break;
    h, w = framer.shape[:2]
    newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
    newimg = cv2.undistort(framer, K, d, None, newcamera)
    resize2 = cv2.resize(newimg, (640, 360))
    # save image
    # ESC 키누르면 종료
    cv2.imshow("dfdfd",newimg)
    cv2.imshow("dfdf2d",resize2)
    if cv2.waitKey(1) & 0xFF == 27:
        break


capr.release()
cv2.destroyAllWindows()

