import math
import matplotlib.pyplot as plt

q1=10
q2=10
l1=3
l2=4

de_q1 = math.radians(q1)
de_q2 = math.radians(q2)

axis_x=[0]
axis_y=[0]

for i in range(1,10):
    de_q1=de_q1+math.radians(10)
    de_q2=de_q2+math.radians(10)
    x1 = l1 * math.cos(de_q1)
    x2 = l1 * math.cos(de_q1) + l2 * math.cos(de_q1 + de_q2)
    y1 = l1 * math.sin(de_q1)
    y2 = l1 * math.sin(de_q1) + l2 * math.sin(de_q1 + de_q2)
    axis_x.append(x1)
    axis_x.append(x2)
    axis_y.append(y1)
    axis_y.append(y2)
    plt.axis([-10,10,-10,10])
    plt.plot(axis_x,axis_y)
    plt.draw()
    plt.pause(1)
    plt.cla()
    axis_x=[0]
    axis_y=[0]
plt.show