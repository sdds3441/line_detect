import cv2
import numpy as np
import re
i=0

coor="C:/Users/201921343/Desktop/0249/1.txt"
with open(coor) as data:
    lines=data.readlines()

numbers = []
num0=[]
num1=[]
num2=[]
num3=[]

for line in lines:
    numbers.append(line[8:-2].split(')('))

for q in range(len(numbers)):
    globals()['num{}'.format(q)].append(1)

print(num0)

for e in numbers[1]:
    # print(e)
    sav = e.split(',')
    num1.append([int(sav[0]), int(sav[1])])

for e in numbers[2]:
    # print(e)
    sav = e.split(',')
    num2.append([int(sav[0]), int(sav[1])])

for e in numbers[3]:
    # print(e)
    sav = e.split(',')
    num3.append([int(sav[0]), int(sav[1])])

fr = len(num0) - 1
se = len(num1) - 1
th = len(num2) - 1
fo = len(num3) - 1

linef = cv2.line(lowf, (num0[0][0], num0[0][1]), (num0[fr][0], num0[fr][1]), (255, 0, 0), 1)
linef = cv2.line(lowf, (num1[0][0], num1[0][1]), (num1[se][0], num1[se][1]), (0, 255, 0), 1)
linef = cv2.line(lowf, (num2[0][0], num2[0][1]), (num2[th][0], num2[th][1]), (0, 0, 255), 1)
linef = cv2.line(lowf, (num3[0][0], num3[0][1]), (num3[fo][0], num3[fo][1]), (255, 255, 0), 1)

linef = cv2.line(lowf, (globals()['num{}'.format(re)][0][0], globals()['num{}'.format(re)][0][1]),
                 (globals()['num{}'.format(re)][dot_length][0], globals()['num{}'.format(re)][dot_length][1]),
                 (255, 0, 0), 1)

linef = cv2.polylines(lowf, globals()['num{}'.format(re)], False, (255, 0, 0))
