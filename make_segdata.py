import cv2
from os import listdir
import numpy as np

input_data_path = "C:/Users/201921343/Desktop/X_2020.07.30.11.18_left (17)"
label_data_path = "C:/Users/201921343/Desktop/Y_seg"
file_name=listdir(input_data_path)
file_name.sort()
file_list_image = [f for f in file_name]
count=1402
for file in file_list_image:
    img=cv2.imread(input_data_path+'/'+file)
    #print(file)
    #result = int(filter(str.isdigit, file))
    img_b,img_g,img_r = cv2.split(img)
    ret, img_r = cv2.threshold(img_r, 254, 255, cv2.THRESH_BINARY)
    ret, img_g = cv2.threshold(img_g, 254, 255, cv2.THRESH_BINARY)
    ret, img_b = cv2.threshold(img_b, 254, 255, cv2.THRESH_BINARY)

    img = cv2.merge([img_b, img_g, img_r])
    print(file)
    cv2.imshow('lane', img)
    cv2.imwrite(label_data_path+'/17%05d'%(count)+'.png', img)
    #cv2.waitKey(1)

    count=count+1
    #if count == 774:
       # count=775
    '''if count == 147:
        count=156
    if count == 421:
        count=428
    if count == 585:
        count=595
    if count == 697:
        count=705
    if count == 947:
        count=953
    if count == 1209:
        count=1221'''

