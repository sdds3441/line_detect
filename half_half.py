import cv2
import numpy as np

capr = cv2.VideoCapture('C:/Users/201921343/Desktop/safdad/outputr.mp4')
capl = cv2.VideoCapture('C:/Users/201921343/Desktop/safdad/outputl.mp4')

while (capr.isOpened()):

    ret, framer = capr.read()
    ret, framel = capl.read()

    framer = cv2.resize(framer, (720, 540))
    length,width,channel=framer.shape
    h_width = int(width / 2)
    dstr = framer.copy()
    dstr = framer[0:length, 270:width]

    framel = cv2.resize(framel, (720, 540))
    dstl = framel.copy()
    dstl = framel[0:length, 0:450]
    cv2.waitKey(30)


    img=cv2.hconcat([dstl,dstr])

    cv2.imshow('plus',img)
   # cv2.imshow('pdlus',framer)

cv2.destroyAllWindows()
