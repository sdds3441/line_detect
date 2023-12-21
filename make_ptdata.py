import cv2
from os import listdir
import numpy as np

lane_color = [[0,0,255],[0,255,0],[255,0,0],[255,0,255],[0,255,255],[255,255,0]]

input_data_path = "C:/Users/201921343/Desktop/X_seg"
label_data_path = "C:/Users/201921343/Desktop/X_pt"
file_name=listdir(input_data_path)
file_name.sort()
file_list_image = [f for f in file_name]



for file in file_list_image:
    img=cv2.imread(input_data_path+'/'+file)
    imgshape=np.shape(img)
    f = open(label_data_path+"/"+file[:len(file)-4]+".txt", 'w')


    lane = []
    for i in range(50):
        
        color1 = -1
        color2 = 2
        color3 = -1
        color4 = 2
        color5 = -1
        color6 = 2
        img_wide=img[int((100-i*2)*0.01*imgshape[0])-1].copy()

        trigger1=0
        trigger2=0
        
        for j in range(imgshape[1]):
            if img_wide[j][0] == lane_color[0][0] and img_wide[j][1] == lane_color[0][1] and img_wide[j][2] == lane_color[0][2]:
                color1 = j/imgshape[1]
            elif img_wide[j][0] == lane_color[1][0] and img_wide[j][1] == lane_color[1][1] and img_wide[j][2] == lane_color[1][2]:
                color2 = j/imgshape[1]
            elif img_wide[j][0] == lane_color[2][0] and img_wide[j][1] == lane_color[2][1] and img_wide[j][2] == lane_color[2][2]:  
                color3 = j/imgshape[1]
            elif img_wide[j][0] == lane_color[3][0] and img_wide[j][1] == lane_color[3][1] and img_wide[j][2] == lane_color[3][2]:
                color4 = j/imgshape[1]
            elif img_wide[j][0] == lane_color[4][0] and img_wide[j][1] == lane_color[4][1] and img_wide[j][2] == lane_color[4][2]:
                color5 = j/imgshape[1]
            elif img_wide[j][0] == lane_color[5][0] and img_wide[j][1] == lane_color[5][1] and img_wide[j][2] == lane_color[5][2]:
                color6 = j/imgshape[1]

        if color5 < 0:
            trigger1=1
        if color6 > 1:
            trigger2=1
            
            
        
        if trigger1 == 0:
            if color1 == -1:
                color1 = color5
            if color5 == -1:
                color5 = color1
        if trigger2 == 0:
            if color2 == 2:
                color2 = color6
            if color6 == 2:
                color6 = color2

        
        lane.append([str(color1),str(color2),str(color3),str(color4),str(color5),str(color6)])
    lane = list(map(list, zip(*lane)))
    for n in range(6):
        f.writelines(",".join(lane[n])+'\n')
    cv2.imshow('lane', img)
    print(file)
    #cv2.imwrite(label_data_path+'/'+file, img)
    f.close()
    cv2.waitKey(1)
