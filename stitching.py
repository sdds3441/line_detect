import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

capr = cv2.VideoCapture('2020.07.25.11.42.r.mp4')
capl = cv2.VideoCapture('2020.07.25.11.42.l.mp4')

capr.set(3,640)
capr.set(4,340)
#capr.set(cv2.CAP_PROP_EXPOSURE,40)

capl.set(3,640)
capl.set(4,340)
#capl.set(cv2.CAP_PROP_EXPOSURE,40)
#path = '.\\images\\'
images = []
while 1:

    ret, framer = capr.read()
    ret, framel = capl.read()

    images.append(framer)
    images.append(framel)

    cv2.imshow("r",images[0])
    cv2.imshow("l",images[1])
    cv2.waitKey()
    mode = cv2.STITCHER_PANORAMA
    # mode = cv2.STITCHER_SCANS

   # if int(cv2.__version__[0]) == 3:
       # stitcher = cv2.createStitcher(mode)
  #  else:
    stitcher = cv2.createStitcher(mode)


    status, stitched = stitcher.stitch(images)

    if status == 0:
        cv2.imwrite(os.path.join('imgs', "IMG_NAME", 'result.jpg'), stitched)

        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    else:
        print('failed... %s' % status)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
#cv2.waitKey(0)print("stitching completed successfully.")

