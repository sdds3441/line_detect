import sympy as sp
sp.init_printing()

conv2Rad = lambda x : x*sp.pi/180

theta1 = sp.Symbol('theta1')
theta2 = sp.Symbol('theta2')
theta3 = sp.Symbol('theta3')
theta4 = sp.Symbol('theta4')
theta5 = sp.Symbol('theta5')
theta6 = sp.Symbol('theta6')

a2, a3 = sp.symbols('a2 a3')
d3, d4 = sp.symbols('d3 d4')

def RotZ(a):
    return sp.Matrix( [  [sp.cos(a), -sp.sin(a), 0, 0],
                        [sp.sin(a), sp.cos(a), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1] ] )

def RotX(a):
    return sp.Matrix( [  [1, 0, 0, 0],
                        [0, sp.cos(a), -sp.sin(a), 0],
                        [0, sp.sin(a), sp.cos(a), 0],
                        [0, 0, 0, 1] ] )

def D_q(a1,a2,a3):
    return sp.Matrix([[1,0,0,a1],[0,1,0,a2],[0,0,1,a3],[0,0,0,1]])

Trans_0to1 = RotZ(theta1)
Trans_1to2 = RotX(conv2Rad(-90))*RotZ(theta2)
Trans_2to3 = D_q(a2,0,0)*D_q(0,0,d3)*RotZ(theta3)
Trans_3to4 = RotX(conv2Rad(-90))*D_q(a3,0,0)*D_q(0,0,d4)*RotZ(theta4)
Trans_4to5 = RotX(conv2Rad(90))*RotZ(theta5)
Trans_5to6 = RotX(conv2Rad(-90))*RotZ(theta6)


Trans_0to2 = sp.simplify(Trans_0to1*Trans_1to2)
Trans_0to3 = sp.simplify(Trans_0to2*Trans_2to3)
Trans_0to4 = sp.simplify(Trans_0to3*Trans_3to4)
Trans_0to5 = sp.simplify(Trans_0to4*Trans_4to5)
Trans_0to6 = sp.simplify(Trans_0to5*Trans_5to6)
sp.pprint(Trans_0to1)