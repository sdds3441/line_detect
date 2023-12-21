import cv2
from os import listdir
import numpy as np

input_data_path = "D:/X_2020.08.20.55.55_left"
label_data_path = "D:/X_2020.08.20.55.55_left"
file_list_image = [f for f in listdir(input_data_path)]
count=1
for file in file_list_image:
    img=cv2.imread(input_data_path+'/'+file)
    img=cv2.resize(img,(1024,576))
    print(file)
    cv2.imshow('lane', img)
    cv2.imwrite(label_data_path+'/7%05d'%(count)+'.png', img)
    cv2.waitKey(1)

    count=count+1
