import matplotlib.pyplot as plt
import math
import cv2 as cv    # OpenCV import
import numpy as np  # 행렬(img)를 만들기 위한 np import
zeroes=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x1=[]
y1=[]
x2=[]
x3=[]
y2=[]
y3=[]
for idx in range(0, 20):
    l1 = 0.5
    l2=0.4
    l3=0.3
    degree = 4.7 * idx
    rad = math.radians(degree)
    new_end_x1 = float(l1 * math.cos(rad))
    new_end_y1 = float(l1 * math.sin(rad))
    new_end_x2 = float(new_end_x1+l2*math.cos(rad*2))
    new_end_y2 = float(new_end_y1+l2*math.sin(rad*2))
    new_end_x3 = float(new_end_x2+l3*math.cos(rad*3))
    new_end_y3 = float(new_end_y2+l3*math.sin(rad*3))
    x1.append(new_end_x1)
    y1.append(new_end_y1)
    x2.append(new_end_x2)
    y2.append(new_end_y2)
    x3.append(new_end_x3)
    y3.append(new_end_y3)
for i in range(0,20):
    plt.plot([0,x1[i]],[0,y1[i]])
    plt.plot([x1[i],x2[i]],[y1[i],y2[i]])
    plt.plot([x2[i], x3[i]], [y2[i], y3[i]])
    plt.grid(True)
    plt.xlim([-1,2])
    plt.ylim([-1,2])
plt.title('3Dof robot arm')
plt.xlabel('x axis(m)')
plt.ylabel('y axis(m)')
plt.show()
