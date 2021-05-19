#!/usr/bin/env python
# coding: utf-8

# # Раздел 6. Уравнения с частными производными параболического типа.
# ## Лабораторная работа №6. Методы решения квазилинейного уравнения теплопроводности.
# ### Вариант 2 Задание 5
# _Яромир Водзяновский_

# Дифференциальная задача
# 
# $$\frac{\partial u}{\partial t} = \frac{\partial}{\partial x} \left( u^{1/3} \frac{\partial u}{\partial x} \right) + \frac{\partial}{\partial y} \left( u^{1/3} \frac{\partial u}{\partial y} \right), \;\;\; 0 < t\leq 1, \; 0 < x,y <1    $$
# $$u(0,x,y) = (1+x+y)^6 / 27000, \; 0 \leq x,y \leq 1  $$
# $$u(t,0,y) = (1+y)^6 / (30-28t)^3, \; 0 < t \leq 1, \; 0 \leq y \leq 1  $$
# $$u(t,1,y) = (2+y)^6/(30-28t)^3, \; 0 < t \leq 1, \; 0 \leq y \leq 1   $$
# $$u(t,x,0) = (1+x)^6 /(30-28t)^3, \; 0 < t \leq 1, \; 0 < x < 1    $$
# $$u(t,x,1) = (2+x)^6 /(30-28t)^3, \; 0 < t \leq 1, \; 0 < x < 1    $$
# Решение в виде: 
# $$u = (C_z + x + y)^{2/\mu} \left[ C_t - \frac{4 (\mu +2)}{\mu}t \right]^{-1/\mu} $$
# 
# Для нашей задачи:
# $$C_t = 30, \; C_z = 1,\; \mu = \frac{1}{3}$$


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from mpl_toolkits.mplot3d import Axes3D



def anal(t,x,y):
    return (C_z + x + y)**(2/nu)*(C_t - 4*t*(nu+2)/nu)**(-1/nu)



def U_anal(x,y,t):   

    t_a = []
    for i in range(N):
        for j in range((L)*(K)):
            t_a.append(t[i])

    y_ar = []
    y_a = np.array([])
    for i in range(K):
        for j in range(L):
            y_ar.append(y[i])
    y_ar = np.array(y_ar)
    for n in range(N):
        y_a = np.concatenate((y_a,y_ar))

    x_a = np.array([])
    for i in range((K)*(N)):
        x_a = np.concatenate((x_a,x))

    U_an = np.vectorize(anal)(x_a,y_a,t_a).reshape(N,L,K)
    return U_an



def run(U_t, U_t_next):
    U_k = U_t
    u_next = np.empty(shape=(L,M))

    max_diff = eps + 1

    while max_diff > eps:
        u_next = np.empty(shape=(L,M))
        u_next[ : , 0] = U_t_next[: , 0]
        u_next[ : , -1] = U_t_next[: , -1]
        u_next[ 0 , :] = U_t_next[0 , :]
        u_next[ -1 , :] = U_t_next[-1 , :]
        for m in range(1, M-1):
            a = np.array([-dt/(2*hx**2)*((U_k[l+1][m])**nu + (U_k[l][m])**nu) for l in range(1, L-1)])
            c = np.array([-dt/(2*hx**2)*((U_k[l-1][m])**nu + (U_k[l][m])**nu) for l in range(1, L-1)])
            b = -a-c+1
            d = U_t[1:-1, m]
            
            alpha = [-a[0] / b[0]]
            beta = [(d[0] - c[0] * U_t_next[0][m]) / b[0]]
            for l in range(1, L - 2):
                alpha.append(-a[l] / (b[l] + c[l]*alpha[l-1])) #28
                beta.append((d[l] - c[l]*beta[l-1]) / (b[l] + c[l]*alpha[l-1]))
            
            for l in range(L - 2, 0, -1):
                u_next[l][m] = alpha[l - 1] * u_next[l + 1][m] + beta[l - 1]

        for l in range(1, L-1):
            a = np.array([-dt/(2*hy**2)*((U_k[l][m+1])**nu + (U_k[l][m])**nu) for m in range(1, M-1)])
            c = np.array([-dt/(2*hy**2)*((U_k[l][m-1])**nu + (U_k[l][m])**nu) for m in range(1, M-1)])
            b = -a-c+1
            d = u_next[l, 1:-1]

            alpha = [-a[0] / b[0]]
            beta = [(d[0] - c[0] * U_t_next[l][0]) / b[0]]

            for l in range(1, M - 2):
                alpha.append(-a[l] / (b[l] + c[l]*alpha[l-1]))
                beta.append((d[l] - c[l]*beta[l-1]) / (b[l] + c[l]*alpha[l-1]))

            for m in range(M - 2, 0, -1):
                u_next[l][m] = alpha[m - 1] * u_next[l][m + 1] + beta[m - 1]

        max_diff = np.max(np.abs((u_next - U_k)[1:-1] / u_next[1:-1]))
        U_k = u_next
        
    return U_k

print('Введи N:')
N = int(input())
print('Введи K:')
M = int(input())
print('Введи L:')
L = int(input())

C_z = 1
C_t = 30
nu = 1/3

###
delta = 0.0
eps = 0.0001
# N = 10
# L = 10
# M = 10
###
xl, hx = np.linspace(0, 1, L, retstep=True)
ym, hy = np.linspace(0, 1, M, retstep=True)
tn, dt = np.linspace(0, 1, N, retstep=True)

# np.set_printoptions(formatter={'all':lambda x: np.format_float_scientific(x, precision = 2)})
pd.set_option('display.float_format', lambda x: '{:.3E}'.format(x))



U = np.zeros(shape=(N, L, M))
U[0] = (C_z + xl + ym)**(2/nu)*C_t**(-1/nu)
for n in range(0, N):
    U[n, 0, :] = (C_z + ym)**(2/nu)*(C_t - 4*(nu+2)*tn[n]/nu)**(-1/nu)
    U[n, -1, :] = (C_z + 1 + ym)**(2/nu)*(C_t - 4*(nu+2)*tn[n]/nu)**(-1/nu)
    U[n, :, 0] = (C_z + xl)**(2/nu)*(C_t - 4*(nu+2)*tn[n]/nu)**(-1/nu)
    U[n, :, -1] = (C_z + 1 + xl)**(2/nu)*(C_t - 4*(nu+2)*tn[n]/nu)**(-1/nu)

for n, t in enumerate(tn[1:], 1):
    U[n] = run(U[n-1],U[n])
u_a = anal(tn[-1], xl, ym[1])
u = U[-1, :, 1]

step_for_output = (L - 1) // 5

numeric = pd.DataFrame(columns = ym[::step_for_output], index = xl[::step_for_output])
numeric.iloc[:, :] = U[-1, ::step_for_output, ::step_for_output]

analitic = pd.DataFrame(columns = ym[::step_for_output], index = xl[::step_for_output])
a=0
for i in range(0, L, step_for_output): #L=M
    analitic.iloc[:, a] = anal(1, xl[::step_for_output], ym[i])
    a = a + 1

difference = np.abs(numeric.to_numpy() - analitic.to_numpy())
dif =  pd.DataFrame(difference, columns = ym[::step_for_output], index = xl[::step_for_output])

max_dif = np.max(np.abs(numeric.to_numpy() - analitic.to_numpy()))
    
    


print('Аналитическое')
print(analitic)


print('Численное')


print(numeric)


print('Разница')
print(dif )


print('Максимальная разница')

print(max_dif)


# ## 3-х мерный график при t = 1


fig = plt.figure(figsize = (15,10))
ax = fig.add_subplot(111, projection='3d')
xgrid, ygrid = np.meshgrid(xl, ym)
ax.plot_surface(ygrid, xgrid, U[N-1])
ax.set_zlabel('U')
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_title('График численного решения')
plt.show()




