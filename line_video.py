import numpy as np
import cv2
import pickle
import time

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def undistort(img, cal_dir='C:\\Users\\201921343\\Desktop\\cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    start = time.time()
    img = undistort(img)
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    print("pipe",time.time()-start)
    return combined_binary

def perspective_warp(img,
                     dst_size=(1280,960),
                     src=np.float32([(0.33,0.47),(0.78,0.47),(0.18,0.7),(1,0.7)]),
                     dst=np.float32([(0.1,0.1), (0.9, 0.1), (0.1,0.9), (0.9,0.9)])):
    start = time.time()
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    print("warp", time.time() - start)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]


def sliding_window(img, nwindows=15, margin=150, minpix=1, draw_windows=True):
    start = time.time()
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

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    print("sliding", time.time() - start)

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), (dotted_lane_left, dotted_lane_right)

def hsv_img(cap):
    start = time.time()
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
    print("hsv", time.time() - start)

    return res1, res2, res3

def draw_lanes(img, left_fit, right_fit, dotted):
    start = time.time()
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_img = np.zeros_like(img)
    right_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    #points = np.hstack((left, right))

    #cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
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
    print("draw", time.time() - start)
    return inv_perspective

def Right_line_color(white_img):
    start = time.time()
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

    lane_color = 100

    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    print("right", time.time() - start)
    return lane_color, test
def Left_line_color(color_img):
    start = time.time()
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

    lane_color = 100

    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    print("left", time.time() - start)
    return lane_color, test

def Find_lane(img):
    start = time.time()
    imgz = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlap_img = pipeline(imgz)
    warpped_img = perspective_warp(overlap_img)
    yellow_img, blue_img, white_img = hsv_img(perspective_warp(undistort(img)))
    sliding_window_img, curves, lanes, dotted_lane = sliding_window(warpped_img)
    print("find", time.time() - start)
    return imgz, curves, dotted_lane, yellow_img, blue_img, white_img

img = cv2.VideoCapture('C:\\Users\\201921343\\Desktop\\videoes\\kcity201908011624.mp4')
width = 1280
height = 960
None_img = np.zeros((height, width, 3), np.uint8)

while(1):
    jkl, imgz = img.read()
    if jkl == False:
        break
    '''
    pt1 = (430, 460)
    pt2 = (1000, 460)
    pt3 = (240, 672)
    pt4 = (1290, 672)
    cv2.circle(imgz, pt1, 10, (255, 0, 0), -1)
    cv2.circle(imgz, pt2, 10, (0, 255, 0), -1)
    cv2.circle(imgz, pt3, 10, (0, 0, 255), -1)
    cv2.circle(imgz, pt4, 10, (255, 255, 255), -1)'''

    img_, lane, dot, yellow_img, blue_img, white_img = Find_lane(imgz)
    asd = draw_lanes(None_img, lane[0], lane[1], dot)
    cv2.imshow('sdfg', imgz)
    imga = weighted_img(yellow_img, white_img)
    cv2.imshow('sbvc', imga)

    #new_img = white_img + yellow_img
    #new_img = new_img + blue_img
    #R_line, R_test = Right_line_color(new_img)
    #L_line, L_test = Left_line_color(new_img)
    #cv2.imshow('L_Color', L_test)
    #cv2.imshow('R_Color', R_test)
    cv2.imshow('lane', img_)
    k = cv2.waitKey(1) & 0xFF
    #print(L_line, R_line)

    if k == 27:
        break

cv2.destroyAllWindows()