import numpy as np
import cv2
import pickle
import time

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def hsv_img(cap):
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 50, 255])

    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_white, upper_white)

    res1 = cv2.bitwise_and(cap, cap, mask=mask1)
    res2 = cv2.bitwise_and(cap, cap, mask=mask2)
    res3 = cv2.bitwise_and(cap, cap, mask=mask3)
    res=res1+res3+res2
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    return res1, res2, res3,dst

def perspective_warp(img,
                     dst_size=(1280,960),
                     src=np.float32([(0.33,0.47),(0.78,0.47),(0.18,0.7),(1,0.7)]),
                     dst=np.float32([(0.2,0.1), (0.8, 0.1), (0.2,0.9), (0.8,0.9)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]


def sliding_window(img, nwindows=10, margin=40, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    dotted_lane_l = 0
    dotted_lane_r = 0

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            dotted_lane_l = dotted_lane_l + 1
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            dotted_lane_r = dotted_lane_r + 1

    if dotted_lane_l >= 1:
        dotted_lane_left = True
    else:
        dotted_lane_left = False
    if dotted_lane_r >= 1:
        dotted_lane_right = True
    else:
        dotted_lane_right = False

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if lefty.size == 0:
        print("왼쪽 차선이 비었습니다.")
        left_fit = np.polyfit(righty, rightx-800, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])
    else:
        left_fit = np.polyfit(lefty, leftx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

    if righty.size == 0:
        print("오른쪽 차선이 비었습니다.")
        right_fit = np.polyfit(lefty, leftx+800, 2)

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])
    else:
        right_fit = np.polyfit(righty, rightx, 2)

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), (dotted_lane_left, dotted_lane_right)


def draw_lanes(img, left_fit, right_fit, dotted):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_img = np.zeros_like(img)
    right_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])

    if dotted[0] == True:
        left_color = (250, 0, 0)
    else:
        left_color = (0, 255, 153)
    if dotted[1] == True:
        right_color = (250, 0, 0)
    else:
        right_color = (0, 255, 153)
    cv2.polylines(left_img, np.int_(left), False, left_color, 30)
    cv2.polylines(right_img, np.int_(right), False, right_color, 30)
    color_img = weighted_img(left_img, right_img)
    inv_perspective = cv2.addWeighted(img, 1, color_img, 0.7, 0)
    return inv_perspective

def Right_line_color(white_img):
    test = white_img[0:white_img.shape[0], int(lane[-1][0]):int(lane[-1][-1]) + 10]
    blue_color = sum(np.transpose(np.reshape(test, (-1, 3)))[0])
    green_color = sum(np.transpose(np.reshape(test, (-1, 3)))[1])
    red_color = sum(np.transpose(np.reshape(test, (-1, 3)))[2])
    if blue_color < green_color:
        bg = blue_color / green_color * 100
    else:
        bg = -green_color / blue_color * 100

    if blue_color < red_color:
        br = blue_color / red_color * 100
    else:
        br = -red_color / blue_color * 100


    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    return lane_color, test
def Left_line_color(color_img):
    test = color_img[0:color_img.shape[0], int(lane[0][0]):int(lane[0][-1]) + 10]
    blue_color = sum(np.transpose(np.reshape(test, (-1, 3)))[0])
    green_color = sum(np.transpose(np.reshape(test, (-1, 3)))[1])
    red_color = sum(np.transpose(np.reshape(test, (-1, 3)))[2])

    if blue_color < green_color:
        bg = blue_color / green_color * 100
    else:
        bg = -green_color / blue_color * 100

    if blue_color < red_color:
        br = blue_color / red_color * 100
    else:
        br = -red_color / blue_color * 100



    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    return lane_color, test

def Find_lane(img):
    start = time.time()
    yellow_img, blue_img, white_img,dst = hsv_img(perspective_warp(img))
    sliding_window_img, curves, lanes, dotted_lane = sliding_window(dst)
    print("find", time.time() - start)
    return sliding_window_img, curves, dotted_lane, yellow_img, blue_img, white_img

img = cv2.VideoCapture('C:\\Users\\201921343\\Desktop\\videoes\\cut.mp4')
width = 1280
height = 960
None_img = np.zeros((height, width, 3), np.uint8)

while(1):
    jkl, imgz = img.read()
    if jkl == False:
        break

    img_, lane, dot, yellow_img, blue_img, white_img = Find_lane(imgz)
    asd = draw_lanes(None_img, lane[0], lane[1], dot)
    cv2.imshow('lane', img_)
    cv2.imshow('sdfg', asd)
    imga = weighted_img(yellow_img, white_img)
    #cv2.imshow('sbvc', imga)

    k = cv2.waitKey(1) & 0xFF


    if k == 27:
        break

cv2.destroyAllWindows()