import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from matplotlib import animation as ani
# src = np.array([[400, 130], [600, 130], [532, 576], [492, 576]], dtype=np.float32)
# dst = np.array([[402, 40], [622, 40], [514, 1024], [510, 1024]], dtype=np.float32)
# src = np.array([[590, 400], [680, 400], [652, 960], [628, 960]], dtype=np.float32)
# dst = np.array([[385, 50], [415, 50], [401, 800], [399, 800]], dtype=np.float32)
src = np.array([(490, 350),(810, 350),(0, 700),(1270, 700)], dtype=np.float32)
dst = np.array([[0, 0], [1280, 0], [0, 960], [1280, 960]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src, dst)
M2 = cv2.getPerspectiveTransform(dst, src)

size = (1280, 960)

def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 이미지를 1:1비율로 OverLap하는 함수
   return cv2.addWeighted(initial_img, α, img, β, λ)

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

lower_W_line = np.array([0, 0, 120])
upper_W_line = np.array([180, 50, 240])

lower_Y_line = np.array([10, 50, 50])
upper_Y_line = np.array([30, 255, 255])
histap=[]
def hsv_img(frame_warp):  # hsv색상표를 이용해 흰색, 노란색, 파란색을 검출하는 함수
   hsv = cv2.cvtColor(frame_warp, cv2.COLOR_BGR2HSV)

   Y_line_range = cv2.inRange(hsv, lower_Y_line, upper_Y_line)
   Y_line_result = cv2.bitwise_and(frame_warp, frame_warp, mask=Y_line_range)

   W_line_range = cv2.inRange(hsv, lower_W_line, upper_W_line)
   W_line_result = cv2.bitwise_and(frame_warp, frame_warp, mask=W_line_range)

   frame_threshold = W_line_result + Y_line_result

   frame_gray1 = cv2.cvtColor(frame_threshold, cv2.COLOR_BGR2GRAY)

   kernel = np.ones((3, 3), np.uint8)
   frame_gray1 = cv2.dilate(frame_gray1, kernel, iterations=1)

   frame_gray2 = cv2.cvtColor(W_line_result, cv2.COLOR_BGR2GRAY)
   cv2.imshow("df",frame_gray2)
   kernel = np.ones((3, 3), np.uint8)
   frame_gray2 = cv2.dilate(frame_gray2, kernel, iterations=1)

   # gray = cv2.cvtColor(W_line_result, cv2.COLOR_RGB2GRAY)
   ret, white_img = cv2.threshold(frame_gray1, 80, 255, cv2.THRESH_BINARY)

   #cv2.imshow('frame_gray', white_img)
   print(np.shape(white_img))
   return frame_gray1, white_img

def perspective_warp(img, dst_size=(1024, 576)):  # 이미지를 버드뷰로 변환
   # cv2.circle(img, (src[0][0], src[0][1]), 2, (0, 255, 0), -1)  # 좌상
   # cv2.circle(img, (src[1][0], src[1][1]), 2, (0, 255, 255), -1)  # 우상
   # cv2.circle(img, (src[2][0], src[2][1]), 2, (255, 255, 0), -1)  # 우하
   # cv2.circle(img, (src[3][0], src[3][1]), 2, (255, 0, 255), -1)  # 좌하
   # img_size = np.float32([(img.shape[1], img.shape[0])])
   # src = src* img_size
   # dst = dst * np.float32(dst_size)
   warped = cv2.warpPerspective(img, M, size)
   # cv2.imshow('dfdf',img)
   return warped

#def get_hist(img):      #이미지의 히스토그램
   #hist = np.sum(img[img.shape[0]//2:,:], axis=0)
   #for a in range(400,800):
       #histap.append(hist[a])
   #bb=sum(histap)
   #if bb>1000000000 and bb<1300000000:
   # print("정지선")
   #return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def stop_line(img, i, nwindows = 2, margin=400, minpix=9000, draw_windows=True):
   center_fit = np.empty(3)
   img_shape = np.shape(img)

   None_img = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
   out_img = np.dstack((img, img, img)) * 255
   center = img_shape[1] // 2

   window_height = np.int((img.shape[0]) // 50)
   nonzero = img.nonzero()
   nonzeroy = np.array(nonzero[0])
   nonzerox = np.array(nonzero[1])
   x_current = center

   stop_lane_inds = []

   count = 0

   for window in range(nwindows, 0, -1):
      win_y_low = window * window_height + i * window_height #img.shape[0] // 2 +
      win_y_high = (window + 1) * window_height + i * window_height #img.shape[0] // 2 +
      win_x_low = x_current - margin
      win_x_high = x_current + margin


      if draw_windows == True:
         cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (100, 255, 255), 2)
      good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
      win_y_mid=(win_y_high-win_y_low)+win_y_low
      stop_lane_inds.append(good_inds)

      left_color = (250, 0, 0)

      if len(good_inds) > minpix:
         x_current = np.int(np.mean(nonzerox[good_inds]))

         point1=np.array([[win_x_low,win_y_mid],[win_x_high,win_y_mid]])

         count += 2

      if count > 1:
         stop = True
      else:
         stop = False
      inv_perspective = None_img

   if len(good_inds) > minpix:
      cv2.polylines(None_img, [point1], False, left_color, 10)

   stop_lane_inds = np.concatenate(stop_lane_inds)
   out_img[nonzeroy[stop_lane_inds], nonzerox[stop_lane_inds]] = [255, 0, 100]

   return out_img, inv_perspective, stop

def draw_lanes(img, left_fit, right_fit, dotted):       #이미지에 차선을 그리는 함수
   ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
   left_img = np.zeros_like(img)
   right_img = np.zeros_like(img)

   left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
   right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])

   if dotted[0] == True:
      left_color = (255, 255, 0)
   else:
      left_color = (255, 255, 0)
   if dotted[1] == True:
      right_color = (255, 255, 0)
   else:
      right_color = (255, 255, 0)
   cv2.polylines(left_img, np.int_(left), False, left_color, 3)
   cv2.polylines(right_img, np.int_(right), False, right_color, 3)
   inv_perspective = cv2.addWeighted(right_img, 1, left_img, 1, 0)
   #inv_perspective = cv2.addWeighted(img, 1, color_img, 0.7, 0)
   return inv_perspective

def Find_lane(img, i):
   color_img, white_img = hsv_img(perspective_warp(img))
   # cv2.imshow('test', color_img)
   stop_line_sliding_window_img, draw_stop_line, stop = stop_line(white_img, i)
   #cv2.imshow("",white_img)

   return draw_stop_line, stop, stop_line_sliding_window_img, white_img

capl = cv2.VideoCapture("C:/Users/201921343/Desktop/videoes/kcity_curve.mp4")
#capr = cv2.VideoCapture(0)

width = 1280
height = 960
None_img = np.zeros((height, width, 3), np.uint8)
i = 0

while(1):
   start = time.time()
   jkl, L_img = capl.read()
   #jkl, R_img = capr.read()
   if jkl == False:
      break

   #imgz = cv2.hconcat([L_img, R_img])
   #imgz = cv2.resize(imgz, (1280, 960))

   draw_stop_line, stop, img_stop, white_img = Find_lane(L_img, i)
   #asd = draw_lanes(None_img, lane[0], lane[1], dotted_lane)
#   plt.plot(get_hist(white_img))
  # plt.draw()
   #plt.show()
  # plt.pause(0.01)
  # plt.cla()
   p_img=perspective_warp(L_img)
   p2=cv2.resize(p_img, (480,360))
   img_stop=cv2.resize(img_stop, (480,360))
   cv2.imshow('img', p2)
   cv2.imshow('lanes', img_stop)

   #plt.pause(0.1)
   # plt.close(all)
   #print("find", time.time() - start)

   if stop == True:
      i += 1
      if i > 47:
         i = 0

   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break

cv2.destroyAllWindows()