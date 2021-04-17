#!/usr/bin/env python
# coding: utf-8

# # Раздел 5. Уравнения с частными производными гиперболического типа.
# ## Лабораторная рабода №5. Методы решения уравнения переноса. 
# ### Вариант 3, задание 9.
# _Яромир Водзяновский_

# ## Аналитическое решение 
# Дифференциальная задача:
# $$ \frac{\partial u}{\partial t} - 4\frac{\partial u}{\partial x} = x; \;\;\; 0 < t \leq 1,\; 0 \leq x \leq 1 $$
# $$u(x,0) = \sin{x} - 0.125x^2; \;\; u(1,t) = \sin{(1+4t)} - 0.125$$
# 
# Аналитическое решение:
# $$u(x,t) = \sin{(4t+x)} - \frac{x^2}{8}$$

# ## Численное решение
# Разностная схема:
# $$D_h = \left\{ (x_l,t^n): x_l = hl, hL = 1, l = \overline{0,L};\; t^n = n\tau, \tau N=1,n = \overline{0,N} \right \} $$
# 
# $$u_l^{n+1} = u_l^n + \frac{2\tau}{3h} \left( 2u_{l+3}^n - 9u_{l+2}^n + 18u_{l+1}^n - 11u_l^n \right) + \frac{8\tau^2}{h^2}  \left( -u_{l+3}^n +4u_{l+2}^n -5u_{l+1}^n +2u_l^n  \right) + \frac{32\tau^3}{3h^3} \left(u_{l+3}^n -3u_{l+2}^n +3u_{l+1}^n -u_l^n  \right) + \tau x_l + 2\tau^2, \;\;\;\;\;\;\;\; l = \overline{0,L-3}, n = \overline{0,N-1}$$
# 
# $$u_l^0 = \sin{x_l} - 0.125x_l^2 = \varphi_l, \; l = \overline{0,L}; \;\; u_L^n = \sin{(1+4t^n)} -0.125 = \psi^n,\;n=\overline{1,N}  $$
# 
# $$u_{L-1}^n = ?,\; n = \overline{1,N}; \;\;\; u_{L-2}^n =?,\; n = \overline{1,N}. $$
# 
# Поиск $u_{L-1}^n, \; u_{L-2}^n$:
# 
# $$[u]_{L-1}^n = [u]_L^n - [u'_x]_L^n h - [u''_{xx}]_L^n \frac{h^2}{2} - [u'''_{xxx}]_L^n \frac{h^3}{6}$$
# $$[u]_{L-2}^n = [u]_L^n - [u'_x]_L^n (2h) - [u''_{xx}]_L^n \frac{(2h)^2}{2} - [u'''_{xxx}]_L^n \frac{(2h)^3}{6}$$
# 
# Выразим производные по координате через производные по времени:
# 
# $$[u'_x]_L^n = \frac{1}{a_L^n} \left( -(\dot \psi_t)^n+ b_L^n \right), \;\;\;[u''_x]_L^n = \frac{1}{(a_L^n)^2} \left \{ (\ddot \psi_{tt})^n + a_L^n(b'_x)_L^n \right \}, \;\;\; [u'''_{xxx}]_L^n = -\frac{1}{(a_L^n)^3}(\dddot \psi_{ttt})^n,$$
# 
# где $a_L^n = -4, \;\; b_L^n = x_L$
# 
# Условие спектральной устойчивости:
# $$\tau \leq \frac{2h}{a}$$
# В нашем случае число Куранта 
# $$K = \tau/ h \leq 0.5$$

# In[192]:


import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[206]:


def anal(x,t):
    return np.sin(4*t + x) - x**2 / 8


# In[550]:


L = 20
K = 0.25
N = int(L/K)
h = 1/L
tau = 1/N
a = -4


# In[551]:


x = np.arange(0,1+1/(10*L),1/L)
t = np.arange(0,1+1/(10*N),1/N)


# In[552]:


condition = abs(2*h/a)
def check():
    if tau <= condition:
        return 'Устойчиво'
    else:
        return 'Не устойчиво'
check()


# In[553]:


psi = np.sin(1+4*t) - 0.125
psi_der1 = 4*np.cos(1+4*t)
psi_der2 = -16*np.sin(1+4*t)
psi_der3 = -64*np.cos(1+4*t)
b = x
b_der1 = 1
phi = np.sin(x)-0.125*x**2

u_der1 =  1/a * (-psi_der1 + b[L])
           
u_der2 = 1/a**2 * (psi_der2 + a)

u_der3 = -1/a**3 * psi_der3
pd.set_option('display.float_format', lambda x: '{:.5E}'.format(x))


# In[554]:


t_a = []
for i in range(N+1):
    for j in range(L+1):
        t_a.append(t[i])
t_a = np.array(t_a)

x_a = np.array([])
for i in range(N+1):
    x_a = np.concatenate((x_a,x))

U_an = np.vectorize(anal)(x_a,t_a).reshape((N+1,L+1))   
U_anal = pd.DataFrame(np.vectorize(anal)(x_a,t_a).reshape((N+1,L+1)), columns = x)


# In[555]:


U = np.zeros((N+1,L+1))
U[0] = phi
U.reshape((1,(N+1)*(L+1)))[0][L::L+1] = psi

U_num = pd.DataFrame(U, columns = x)


# In[556]:


h_arr = [h*i for i in range(0,3)]
for j in range(1,3):
    k = L-j
    for i in range(1,N+1):
        
        U[i][k] = U[i][k+1] - u_der1[i] * h_arr[j] - u_der2[i] * h_arr[j]**2 / 2 - u_der3[i] * h_arr[j]**3 / 6


# In[561]:


for n in range(N):
    for l in range(L-2):
        U[n+1][l] = U[n][l] + (2*tau/(3*h)*(2*U[n][l+3] - 9*U[n][l+2] + 18*U[n][l+1] - 11*U[n][l]) + 
                           8*(tau**2)/(h**2)*(-U[n][l+3] +4*U[n][l+2] -5*U[n][l+1] +2*U[n][l]) + 
                           32*(tau**3)/(3*h**3)*(U[n][l+3] - 3*U[n][l+2] + 3*U[n][l+1] - U[n][l]) + 
                           tau*x[l] + 2*tau**2)
ans = pd.DataFrame(U, columns = x)
ans


# In[558]:


from scipy.spatial.distance import cosine

cosine(np.vectorize(anal)(x_a,t_a), U.reshape((1,(N+1)*(L+1))))


# In[559]:


data = np.concatenate((x,U_an[N],U[N], abs(U_an[N]-U[N]), [max(abs(U_an[N]-U[N])) for i in range(L+1)]))
data = data.reshape((5,L+1))
table = pd.DataFrame(data.T, columns = ['x', 'Analitic', 'Numeric', 'Diff', 'Max Diff'])
table


# In[560]:


plt.figure(figsize = (10,6))
plt.plot(x,np.vectorize(anal)(x,[t[N] for i in range(L+1)]))
plt.plot(x,U[N])
plt.legend(['analitic','numeric'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[404]:


# L = 40
# K = 0.25
# N = int(L/K)

# x = np.arange(0,1+1/(10*L),1/L)
# t = np.arange(0,1+1/(10*N),1/N)

# t_a = []
# for i in range(N+1):
#     for j in range(N+1):
#         t_a.append(t[i])
# t_a = np.array(t_a)

# x_a = np.array([])
# for i in range(N+1):
#     x_a = np.concatenate((x_a,x))    

   
# fig = plt.figure(figsize = (15,10))
# ax = fig.add_subplot(111, projection='3d')
# xgrid, tgrid = np.meshgrid(x, t)
# ax.plot_surface(tgrid, xgrid, anal(xgrid,tgrid))
# ax.set_zlabel('U')
# ax.set_ylabel('x')
# ax.set_xlabel('t')
# ax.set_title('График аналитического решения')
# plt.savefig('surface.png', dpi=400, quality=100)


# In[ ]:




