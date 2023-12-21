import numpy as np
import cv2
import time

# 카메라에 접근하기 위해 VideoCapture 객체를 생성
capl = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capr = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# capc = cv2.VideoCapture(3, cv2.CAP_DSHOW)

capl.set(3, 1080)
capl.set(4, 1280)
capl.set(15, -100)
capr.set(3, 1080)
capr.set(4, 1280)
capr.set(15, -100)
# capc.set(3, 1080)
# capc.set(4, 1920)
# capc.set(15, -10)

widthl = int(capl.get(cv2.CAP_PROP_FRAME_WIDTH))
heightl = int(capl.get(cv2.CAP_PROP_FRAME_HEIGHT))
widthr = int(capr.get(cv2.CAP_PROP_FRAME_WIDTH))
heightr = int(capr.get(cv2.CAP_PROP_FRAME_HEIGHT))
# widthc = int(capc.get(cv2.CAP_PROP_FRAME_WIDTH))
# heightc = int(capc.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

writerl = cv2.VideoWriter('outputl.mp4', fourcc, 30.0, (widthl, heightl))
writerr = cv2.VideoWriter('outputr.mp4', fourcc, 30.0, (widthr, heightr))
# writerc = cv2.VideoWriter('outputc.mp4', fourcc, 30.0, (widthc, heightc))

while(True):

    # 이미지를 캡쳐
    ret, framel = capl.read()
    ret, framer = capr.read()
    # ret, framec = capc.read()

    # M = np.ones(framel.shape, dtype="uint8") * 10
    # framel = cv2.subtract(framel, M)
    # # framel = cv2.add(framel, M)
    #
    # M = np.ones(framer.shape, dtype="uint8") * 10
    # framer = cv2.subtract(framer, M)
    # framer = cv2.add(framer, M)

    # M = np.ones(framec.shape, dtype="uint8") * 10
    # framec = cv2.subtract(framec, M)
    # framec = cv2.add(framec, M)

    # 캡쳐되지 않은 경우 처리
    if ret == False:
        break;

    cv2.imshow('testl', framel)
    cv2.imshow('testr', framer)
    # cv2.imshow('testc', framec)

    writerl.write(framel)
    writerr.write(framer)
    # writerc.write(framec)

    # ESC 키누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


capl.release()
capr.release()
# capc.release()
writerl.release()
writerr.release()
# writerc.release()
cv2.destroyAllWindows()