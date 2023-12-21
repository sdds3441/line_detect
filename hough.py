# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def hsv_img(cap):       #hsv색상표를 이용해 흰색, 노란색, 파란색을 검출하는 함수
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([0, 100, 150])       #노란색 최솟값
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

    #gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(res, 30, 255, cv2.THRESH_BINARY)      #이미지를 흑백으로 변경 후 이진화
    #cv2.imshow('3', dst)
    #gray2 = cv2.cvtColor(res3, cv2.COLOR_RGB2GRAY)
    ret, dst2 = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)
    #cv2.imshow('dst',cv2.resize(dst2,(500,1000)))
    return dst, dst2

def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

Left_img = cv2.VideoCapture("C:/Users/201921343/Desktop/2020.07.30.11.18/outputl.mp4")
#Right_img = cv2.VideoCapture(1)
width = 1000
height = 2000
None_img = np.zeros((height, width, 3), np.uint8)

while(1):
    jkl, image = Left_img.read()
    #jkl, R_img = Right_img.read()
    if jkl == False:
        break

    #image = cv2.imread('C:\\Users\\201921343\\Desktop\\solidWhiteCurve.jpg')  # 이미지 읽기
    height, width = image.shape[:2]  # 이미지 높이, 너비

    gg, gray_img = hsv_img(image)  # 흑백이미지로 변환
    ret, dst = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
    #blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

    canny_img = canny(dst, 70, 210)  # Canny edge 알고리즘

    vertices = np.array(
        [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
        dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices)  # ROI 설정

    hough_img = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환


    result = weighted_img(hough_img, image)  # 원본 이미지에 검출된 선 overlap
    #cv2.imshow('result', dst)  # 결과 이미지 출력
    cv2.imshow('result', hough_img)  # 결과 이미지 출력
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


cv2.waitKey(0)