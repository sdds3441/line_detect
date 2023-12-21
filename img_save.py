import cv2
import numpy as np
i=0
cap = cv2.VideoCapture('C:\\Users\\201921343\\Desktop\\IMG_0249.mp4')

while(cap.isOpened()):
    #if i==1:
       # break
    ret, frame = cap.read()

    coor = "C:/Users/201921343/Desktop/0249/"+"%d"%(i+1)+".txt"
    with open(coor) as data:
        lines = data.readlines()

    lowf=frame*0.9
    numbers = []
    num0 = []
    num1 = []
    num2 = []
    num3 = []
    num4 = []
    num5 = []
    num6 = []
    num7 = []
    num8 = []
    num9 = []
    for line in lines:
        numbers.append(line[8:-2].split(')('))

    for re in range(len(numbers)):

        for e in numbers[re]:
        # print(e)
            sav = e.split(',')
            globals()['num{}'.format(re)].append([int(sav[0]), int(sav[1])])
            dot_length=len(globals()['num{}'.format(re)])-1
        linef = cv2.line(lowf, (globals()['num{}'.format(re)][0][0], globals()['num{}'.format(re)][0][1]),(globals()['num{}'.format(re)][dot_length][0], globals()['num{}'.format(re)][dot_length][1]),(255, 0, 0), 1)
        cv2.imwrite('C:/Users/201921343/Desktop/label/'+'%05d'%(i)+'.png', linef)
    #cv2.waitKey()
    print(numbers)
    i+=1
#    coor.close()
cap.release()