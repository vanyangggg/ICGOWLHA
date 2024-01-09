# Import necessary packages
import pandas as pd
import numpy as np
import sklearn.metrics
import math
from sympy import *
from sklearn import preprocessing
from numpy.ma import *
from time import time
from pyomo.environ import *

# Define necessary function

def interval_function(Ymin,Ymax):
    '''
    Ymin:The minimum value of interval data
    Ymax:The maximum value of interval data
    '''
    interval = []
    for i in range(len(Ymax)):
        interval.append([Ymin[i],Ymax[i]])
    return(interval)

def left_metrics_function(pre1_x, pre2_x, pre3_x):  # 单项预测方法的左端点矩阵
    x1_L_metrics = []
    for i in range(len(pre1_x)):
        x1_L_metrics.append(pre1_x[i][0])
    x2_L_metrics = []
    for i in range(len(pre2_x)):
        x2_L_metrics.append(pre2_x[i][0])
    x3_L_metrics = []
    for i in range(len(pre3_x)):
        x3_L_metrics.append(pre3_x[i][0])

    x_L_metrics_T = np.array(x1_L_metrics + x2_L_metrics + x3_L_metrics).reshape(3, len(pre1_x)).T

    return (x_L_metrics_T)

def right_metrics_function(pre1_x, pre2_x, pre3_x):  # 单项预测方法的右端点矩阵
    x1_L_metrics = []
    for i in range(len(pre1_x)):
        x1_L_metrics.append(pre1_x[i][1])
    x2_L_metrics = []
    for i in range(len(pre2_x)):
        x2_L_metrics.append(pre2_x[i][1])
    x3_L_metrics = []
    for i in range(len(pre3_x)):
        x3_L_metrics.append(pre3_x[i][1])

    x_L_metrics_T = np.array(x1_L_metrics + x2_L_metrics + x3_L_metrics).reshape(3, len(pre1_x)).T

    return (x_L_metrics_T)

def CGOWLHA_attitude_function(x_interval, attitude, lamda):
    '''
    x_interval:The aggregated interval data
    attitude:Attitude paremeter
    lamda:Generalized parameter
    '''
    x_L = []
    x_U = []
    for t in range(len(x_interval)):
        x_L.append(x_interval[t][0])
        x_U.append(x_interval[t][1])

    C_pre = list(np.exp(
        (1 / (attitude * ((np.log(x_U)) ** (-lamda)) + ((1 - attitude) * (np.log(x_L)) ** (-lamda)))) ** (1 / lamda)))
    return (C_pre)

def induced_function(real_combine, CGO_pre1, CGO_pre2, CGO_pre3):
    '''
    induced_variable can be calculated by this function
    '''
    e1 = (np.array(real_combine) - np.array(CGO_pre1)) / np.array(real_combine)
    e2 = (np.array(real_combine) - np.array(CGO_pre2)) / np.array(real_combine)
    e3 = (np.array(real_combine) - np.array(CGO_pre3)) / np.array(real_combine)

    u1 = np.zeros(len(real_combine))
    u2 = np.zeros(len(real_combine))
    u3 = np.zeros(len(real_combine))
    for i in range(len(u1)):
        if abs(e1[i]) >= 0 and abs(e1[i]) < 1:
            u1[i] = 1 - abs(e1[i])
        else:
            u1[i] = 0

    for i in range(len(u2)):
        if abs(e2[i]) >= 0 and abs(e2[i]) < 1:
            u2[i] = 1 - abs(e2[i])
        else:
            u2[i] = 0

    for i in range(len(u3)):
        if abs(e3[i]) >= 0 and abs(e3[i]) < 1:
            u3[i] = 1 - abs(e3[i])
        else:
            u3[i] = 0

    u_metrics_T = np.array(list(u1) + list(u2) + list(u3)).reshape(3, len(real_x)).T

    return (u_metrics_T)

def IRMSE_function(ture_interval,predict_interval):
    ture_interval = np.array(ture_interval)
    predict_interval = np.array(predict_interval)
    df = pd.DataFrame(ture_interval-predict_interval)**2
    IRMSE = sum(np.sqrt(df.sum(axis=0)/len(ture_interval)))/2
    return IRMSE

def IMAE_function(ture_interval,predict_interval):
    ture_interval = np.array(ture_interval)
    predict_interval = np.array(predict_interval)
    df = pd.DataFrame(np.abs(ture_interval-predict_interval))
    IMAE = sum(df.sum(axis=0)/len(ture_interval))/2
    return IMAE

def IMSPE_function(ture_interval,predict_interval):
    ture_interval = np.array(ture_interval)
    predict_interval = np.array(predict_interval)
    df = pd.DataFrame((ture_interval-predict_interval)/ture_interval)**2
    IMSPE =  sum(np.sqrt(df.sum(axis=0)/len(ture_interval)))/2
    return IMSPE

def IMAPE_function(ture_interval,predict_interval):
    ture_interval = np.array(ture_interval)
    predict_interval = np.array(predict_interval)
    df = pd.DataFrame(abs((ture_interval-predict_interval)/ture_interval))
    IMAPE = sum(df.sum(axis=0)/len(ture_interval))/2

    return IMAPE

#data obtain preprocess
data = pd.read_csv("data.csv")

Ymax = data["Ymax"]
Ymin = data["Ymin"]
M1_min = data["MLP min"]
M1_max = data["MLP max"]
M2_min = data["Holt min"]
M2_max = data["Holt max"]
M3_min  = data["LSTM min"]
M3_max  = data["LSTM max"]

real_x = interval_function(Ymin=Ymin,Ymax=Ymax)
pre1_x = interval_function(Ymin=M1_min,Ymax=M1_max)
pre2_x = interval_function(Ymin=M2_min,Ymax=M2_max)
pre3_x = interval_function(Ymin=M3_min,Ymax=M3_max)

x_L_metrics_T = left_metrics_function(pre1_x,pre2_x,pre3_x)
x_U_metrics_T = right_metrics_function(pre1_x,pre2_x,pre3_x)

#model solving
path = "ipopt.exe"

real_combine = CGOWLHA_attitude_function(x_interval=real_x, attitude=attitude, lamda=lamda)
real_combine = np.array(real_combine)
b = (sum((np.log(real_combine) ** (-lamda)) ** 2)) ** 0.5
CGO_pre1 = CGOWLHA_attitude_function(x_interval=pre1_x, attitude=attitude, lamda=lamda)
CGO_pre2 = CGOWLHA_attitude_function(x_interval=pre2_x, attitude=attitude, lamda=lamda)
CGO_pre3 = CGOWLHA_attitude_function(x_interval=pre3_x, attitude=attitude, lamda=lamda)
CGO_metrics_T = np.array(CGO_pre1 + CGO_pre2 + CGO_pre3).reshape(3, len(CGO_pre1)).T
CGO_metrics_T = CGO_metrics_T

u_metrics_T = induced_function(real_combine, CGO_pre1, CGO_pre2, CGO_pre3)

x_uindex = []
for t in range(len(real_x)):
    index = np.argsort(u_metrics_T[t])
    for i in range(3):
        x_uindex.append(CGO_metrics_T[t][index[-(int(i + 1))]])

CGO_ordered_metrics = np.array(x_uindex).reshape(len(real_x), 3)

model = ConcreteModel()
model.w1 = Var(domain=Reals, bounds=(0, 1), initialize=0)
model.w2 = Var(domain=Reals, bounds=(0, 1), initialize=0)
model.w3 = Var(domain=Reals, bounds=(0, 1), initialize=1)
w1 = model.w1
w2 = model.w2
w3 = model.w3

pre_combine = []
for i in range(len(real_combine)):
    pre_combine.append(w1 / (pyomo.environ.log(CGO_ordered_metrics[i][0])) ** lamda + w2 / (
        pyomo.environ.log(CGO_ordered_metrics[i][1])) ** lamda + w3 / (
                           pyomo.environ.log(CGO_ordered_metrics[i][2])) ** lamda)

a = (sum((np.log(real_combine) ** (-lamda) - pre_combine) ** 2)) ** 0.5
f = a / b

model.f = Objective(expr=f, sense=minimize)
model.ceq1 = Constraint(expr=w1 + w2 + w3 == 1)
SolverFactory('ipopt', executable=path).solve(model)
print('optimal f: {:.4f}'.format(model.f()))
print('optimal w: [{:.4f}, {:.4f},{:.4f}]'.format(w1(), w2(), w3()))
print("lamda： " + str(lamda))
print("miu：" + str(attitude))
print(model.f())


# obtain the best weights
w = [w1(),w2(),w3()]

#obtain the predicted interval
x_head_L = []
for i in range(len(real_x)):
    x_head_L.append(IGOWLHA_function(u=u_metrics_T[i], x=x_L_metrics_T[i], w=w, lamda=lamda))
x_head_U = []
for i in range(len(real_x)):
    x_head_U.append(IGOWLHA_function(u=u_metrics_T[i], x=x_U_metrics_T[i], w=w, lamda=lamda))

x_combine = []
for i in range(len(x_head_L)):
    x_combine.append([x_head_L[i], x_head_U[i]])

pre_combine = CGOWLHA_attitude_function(x_interval=x_combine, attitude=attitude, lamda=lamda)

#obtain the error metircs
IRMSE1 = IRMSE_function(real_x,pre1_x)
IRMSE2 = IRMSE_function(real_x,pre2_x)
IRMSE3 = IRMSE_function(real_x,pre3_x)
IRMSE_combine = IRMSE_function(real_x,x_combine)
IRMSE_list = [IRMSE1,IRMSE2,IRMSE3,IRMSE_combine]

IMAE1 = IMAE_function(real_x,pre1_x)
IMAE2 = IMAE_function(real_x,pre2_x)
IMAE3 = IMAE_function(real_x,pre3_x)
IMAE_combine = IMAE_function(real_x,x_combine)
IMAE_list = [IMAE1,IMAE2,IMAE3,IMAE_combine]

IMAPE1 = IMAPE_function(real_x,pre1_x)
IMAPE2 = IMAPE_function(real_x,pre2_x)
IMAPE3 = IMAPE_function(real_x,pre3_x)
IMAPE_combine = IMAPE_function(real_x,x_combine)
IMAPE_list = [IMAPE1,IMAPE2,IMAPE3,IMAPE_combine]

IMSPE1 = IMSPE_function(real_x,pre1_x)
IMSPE2 = IMSPE_function(real_x,pre2_x)
IMSPE3 = IMSPE_function(real_x,pre3_x)
IMSPE_combine = IMSPE_function(real_x,x_combine)
IMSPE_list = [IMSPE1,IMSPE2,IMSPE3,IMSPE_combine]

print("IMSE："+str(np.around(IRMSE_list,decimals=5)))
print("IMAE："+str(np.around(IMAE_list,decimals=5)))
print("IMAPE："+str(np.around(IMAPE_list,decimals=5)))
print("IMSPE："+str(np.around(IMSPE_list,decimals=5)))

