import numpy as np
import cv2

capr = cv2.VideoCapture('C:/Users/201921343/Desktop/2020.07.30.11.18/outputr.mp4')
capr.set(3,1080)
capr.set(4,720)
capl=cv2.VideoCapture('C:/Users/201921343/Desktop/2020.07.30.11.18/outputl.mp4')
capl.set(3,1080)
capl.set(4,720)
brightness=0
capr.set(15, brightness)
capl.set(15, brightness)
while (True):
   ret, framer = capr.read()
   ret, framel = capl.read()

   framer_gray = cv2.cvtColor(framer, cv2.COLOR_BGR2GRAY)
   framel_gray = cv2.cvtColor(framel, cv2.COLOR_BGR2GRAY)
   cols, rows = framer_gray.shape
   framer_brightness = np.sum(framer_gray) / (255 * cols * rows)
   framel_brightness = np.sum(framel_gray) / (255 * cols * rows)
   frame_brightness = (framel_brightness + framer_brightness)/2
   print(brightness)

   if (0.55 > frame_brightness and frame_brightness > 0.35) == False:
       brightness = brightness - (frame_brightness - 0.4) * 0.5

       capl.set(15, brightness)
       capr.set(15, brightness)


   #framer = cv2.resize(framer, (720, 480))
   #framel = cv2.resize(framel, (720, 480))

   r_dstPoint = np.array([[467, 355], [512, 355], [512, 576], [351, 576]], dtype=np.float32)
   r_srcPoint = np.array([[512, 355], [557, 355], [673, 576], [512, 576]], dtype=np.float32)


   #l_dstPoint = np.array([[512, 348], [546, 348], [655, 576], [512, 576]], dtype=np.float32)
   #l_srcPoint = np.array([[478, 348], [512, 348], [512, 576], [369, 576]], dtype=np.float32)
   l_srcPoint = np.array([[400, 225], [600, 225], [600, 576], [400, 576]], dtype=np.float32)
   l_dstPoint = np.array([[400, 150], [600, 150], [530, 576], [470, 576]], dtype=np.float32)

   cv2.circle(framel, (l_srcPoint[0][0], l_srcPoint[0][1]), 2, (0, 255, 0),-1)#좌상
   cv2.circle(framel, (l_srcPoint[1][0], l_srcPoint[1][1]), 2, (0, 255, 255),-1)#우상
   cv2.circle(framel, (l_srcPoint[2][0], l_srcPoint[2][1]), 2, (255, 255, 0),-1)#우하
   cv2.circle(framel, (l_srcPoint[3][0], l_srcPoint[3][1]), 2, (255, 0, 255),-1)#좌하
   r_matrix = cv2.getPerspectiveTransform(r_srcPoint, r_dstPoint)
   l_matrix = cv2.getPerspectiveTransform(l_srcPoint, l_dstPoint)
   r_dst = cv2.warpPerspective(framer, r_matrix,(1024,576))
   l_dst = cv2.warpPerspective(framel, l_matrix,(1024,576))


   dstr = r_dst.copy()
   dstr = r_dst[0:576, 512:1024]
   dstl = l_dst.copy()
   dstl = l_dst[0:576, 0:512]

   img = cv2.hconcat([dstl, dstr])
   #cv2.imshow('dfd',r_dst)
   cv2.imshow('dfd',l_dst)
   #cv2.imshow('dst2', dstr)
   #cv2.imshow('dfdd',framer)
   #cv2.imshow('dfddd',framer)
   cv2.imshow('dfdsdd',framel)
  # cv2.imshow('dfdd',img)

   # ESC 키누르면 종료
   if cv2.waitKey(1) & 0xFF == 27:
      break

cv2.destroyAllWindows()