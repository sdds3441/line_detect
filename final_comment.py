import numpy as np
import cv2
import time

def weighted_img(img, initial_img, α=1, β=1., λ=0.):        #이미지를 1:1비율로 OverLap하는 함수
    return cv2.addWeighted(initial_img, α, img, β, λ)

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def hsv_img(cap):       #hsv색상표를 이용해 흰색, 노란색, 파란색을 검출하는 함수
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([0, 50, 150])       #노란색 최솟값
    upper_yellow = np.array([40, 255, 255])     #노란색 최댓값

    lower_blue = np.array([90, 50, 50])         #파란색 최솟값
    upper_blue = np.array([130, 255, 255])      #파란색 최댓값

    lower_white = np.array([0, 0, 230])         #하얀색 최솟값
    upper_white = np.array([180, 50, 255])      #하얀색 최댓값

    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_white, upper_white)

    res1 = cv2.bitwise_and(cap, cap, mask=mask1)
    res2 = cv2.bitwise_and(cap, cap, mask=mask2)
    res3 = cv2.bitwise_and(cap, cap, mask=mask3)

    res=res1+res2+res3      #노란색, 파란색, 하얀색만을 검출한 이미지

    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)      #이미지를 흑백으로 변경 후 이진화

    gray2 = cv2.cvtColor(res3, cv2.COLOR_RGB2GRAY)
    ret, dst2 = cv2.threshold(gray2, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('dst',cv2.resize(dst2,(500,1000)))
    return dst, dst2

def perspective_warp(img,
                     dst_size=(1024,576),
                     src=np.array([[490, 100], [510, 100], [600, 576], [400, 576]], dtype=np.float32),
                     dst=np.array([[450, 0], [550, 0], [520, 2000], [480, 2000]], dtype=np.float32)):        #이미지를 버드뷰로 변환
    #cv2.circle(img, (src[0][0], src[0][1]), 2, (0, 255, 0), -1)  # 좌상
    #cv2.circle(img, (src[1][0], src[1][1]), 2, (0, 255, 255), -1)  # 우상
    #cv2.circle(img, (src[2][0], src[2][1]), 2, (255, 255, 0), -1)  # 우하
    #cv2.circle(img, (src[3][0], src[3][1]), 2, (255, 0, 255), -1)  # 좌하
    img_size = np.float32([(img.shape[1],img.shape[0])])
    #src = src* img_size
    #dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1000,2000))
    #cv2.imshow('dfdf',img)
    return warped

def get_hist(img):      #이미지의 히스토그램
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows = 20, margin = 80, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255      #이미지의 배열 세 개를 겹쳐 쌓은 것에 255를 곱한 것

    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] / 2)                          #히스토그램의 중심 좌표
    leftx_base = np.argmax(histogram[:midpoint])                    #히스토그램의 왼쪽에서 가장 큰 값의 인덱스 출력
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint        #히스토그램의 오른쪽에서 가장 큰 값의 인덱스 출력

    window_height = np.int(img.shape[0] / nwindows)     #윈도우의 높이 설정
    nonzero = img.nonzero()                             #이미지의 0이 아닌 곳의 index를 출력
    nonzeroy = np.array(nonzero[0])                     #차선의 Y값
    nonzerox = np.array(nonzero[1])                     #차선의 X값
    leftx_current = leftx_base                          #히스토그램의 왼쪽 (?)
    rightx_current = rightx_base                        #히스토그램의 오른쪽 (?)

    left_lane_inds = []
    right_lane_inds = []

    dotted_lane_l = 0       #빈 윈도우의 개수 (왼쪽)
    dotted_lane_r = 0       #빈 윈도우의 개수 (오른쪽)

    for window in range(nwindows):
        #윈도우 위아래 위치
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height

        #양쪽의 윈도우 너비 설정
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 양쪽 윈도우 그리기
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)

        #윈도우 안 픽셀 위치를 배열로 생성
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #윈도우 안 픽셀의 유무
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            dotted_lane_l = dotted_lane_l + 1
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            dotted_lane_r = dotted_lane_r + 1

    #빈 윈도우가 1개 이상이면 점선
    if dotted_lane_l >= 1:
        dotted_lane_left = True
    else:
        dotted_lane_left = False
    if dotted_lane_r >= 1:
        dotted_lane_right = True
    else:
        dotted_lane_right = False

    #양쪽 차선들의 배열을 연결
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #한쪽에 차선이 없으면 다른 쪽 차선 모양으로 출력
    if lefty.size == 0:
        print("왼쪽 차선이 비었습니다.")
        dotted_lane_left = False
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
        dotted_lane_right = False
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

    #이차방정식 (?)
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    #????????????
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), (dotted_lane_left, dotted_lane_right)

def stop_line(img, nwindows = 10, margin = 400, minpix = 50000, draw_windows=True):
    global left_a, left_b, left_c
    center_fit = np.empty(3)

    None_img = np.zeros((height, width, 3), np.uint8)
    out_img = np.dstack((img, img, img)) * 255
    center = 600

    window_height = np.int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    x_current = center

    stop_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin


        if draw_windows == True:
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                          (100, 255, 255), 3)
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        win_y_mid=(win_y_high-win_y_low)+win_y_low
        stop_lane_inds.append(good_inds)

        left_color = (250, 0, 0)

        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

            point1=np.array([[win_x_low,win_y_mid],[win_x_high,win_y_mid]])

            cv2.polylines(None_img, [point1], False, left_color, 30)
        stop_lane = None_img

    stop_lane_inds = np.concatenate(stop_lane_inds)

    leftx = nonzerox[stop_lane_inds]
    lefty = nonzeroy[stop_lane_inds]

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    center_fitx = center_fit[0] * ploty ** 2 + center_fit[1] * ploty + center_fit[2]

    out_img[nonzeroy[stop_lane_inds], nonzerox[stop_lane_inds]] = [255, 0, 100]

    return out_img, center_fitx, center_fit, stop_lane

def draw_lanes(img, left_fit, right_fit, dotted):       #이미지에 차선을 그리는 함수
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

def Right_line_color(color_img):        #오른쪽 차선 검출
    #오른쪽 차선 영역
    test = color_img[0:color_img.shape[0], int(lane[-1][0]):int(lane[-1][-1]) + 10]

    #이미지 색 분할
    blue_color = sum(np.transpose(np.reshape(test, (-1, 3)))[0])
    green_color = sum(np.transpose(np.reshape(test, (-1, 3)))[1])
    red_color = sum(np.transpose(np.reshape(test, (-1, 3)))[2])

    #색 비율 계산
    if blue_color < green_color:
        bg = blue_color / green_color * 100
    else:
        bg = -green_color / blue_color * 100

    if blue_color < red_color:
        br = blue_color / red_color * 100
    else:
        br = -red_color / blue_color * 100

    #차선 색 설정
    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    return lane_color, test

def Left_line_color(color_img):         #왼쪽 차선 검출
    #왼쪽 차선 영역
    test = color_img[0:color_img.shape[0], int(lane[0][0]):int(lane[0][-1]) + 10]

    #이미지 색 분할
    blue_color = sum(np.transpose(np.reshape(test, (-1, 3)))[0])
    green_color = sum(np.transpose(np.reshape(test, (-1, 3)))[1])
    red_color = sum(np.transpose(np.reshape(test, (-1, 3)))[2])

    #색 비율 계산
    if blue_color < green_color:
        bg = blue_color / green_color * 100
    else:
        bg = -green_color / blue_color * 100

    if blue_color < red_color:
        br = blue_color / red_color * 100
    else:
        br = -red_color / blue_color * 100

    #차선 색 설정
    if abs(br) > 50 and abs(bg) > 50:
        lane_color = 0  # White
    elif br > 0 and bg > 0:
        lane_color = 1  # Yellow
    else:
        lane_color = 2  # Blue
    return lane_color, test

def Find_lane(img):
    color_img, white_img = hsv_img(perspective_warp(img))
    sliding_window_img, lane, lanes, dotted_lane = sliding_window(color_img)
    #stop_line_sliding_window_img, stop_lane, lanes, draw_stop_line = stop_line(white_img)
    return sliding_window_img, lane, dotted_lane

Left_img = cv2.VideoCapture("C:/Users/201921343/Desktop/2020.07.30.11.18/outputr.mp4")
#Right_img = cv2.VideoCapture(1)
width = 1000
height = 2000
None_img = np.zeros((height, width, 3), np.uint8)

while(1):
    start = time.time()
    jkl, L_img = Left_img.read()
    #jkl, R_img = Right_img.read()
    if jkl == False:
        break

    #imgz = cv2.hconcat([L_img, R_img])
    #imgz = cv2.resize(imgz, (1280, 960))

    img_, lane, dotted_lane = Find_lane(L_img)
    asd = draw_lanes(None_img, lane[0], lane[1], dotted_lane)
    cv2.imshow('ori',L_img)
    per=perspective_warp(L_img)
    #cv2.resize(per,(500,100))
    cv2.imshow('img', cv2.resize(per,(500,1000)))
    #lanes = weighted_img(asd, draw_stop_line)
    cv2.imshow('lanes', cv2.resize(img_,(500,1000)))
    print("find", time.time() - start)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()