import cv2
from os import listdir
import numpy as np

input_data_path = "C:/Users/201921343/Desktop/X_2020.07.30.11.18_left (17)"
label_data_path = "C:/Users/201921343/Desktop/Y_ori"
file_list_image = [f for f in listdir(input_data_path)]
count=1401
#count=60
#count=155
#count=427
#count=594
#count=704
#count=952
#count=1220

img=cv2.imread(input_data_path+'/'+file_list_image[1])
height,width,chan =img.shape
zero = np.zeros((height, width, 1), dtype = np.uint8)
cv2.imwrite(label_data_path+'/17%05d'%(count)+'.png', zero)
