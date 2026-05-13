import matplotlib.pyplot as plt
import numpy as np
from Amp_correction_coefs import def_window

x = np.linspace(0,1,100)
# y = np.linspace(100,1,100)

# x_y = x/y

x_y = np.linspace(0,3, 100)
y = x/x_y

def get_def_window_valmass(x,y,q):
    val_mass = []
    for x_i,y_i in zip(x,y):
        val = def_window(x_i,y_i,q_factor=q)
        val_mass.append(val)
    return val_mass

plt.figure()
plt.plot(x_y,get_def_window_valmass(x,y,1),label='q = 1')
plt.plot(x_y,get_def_window_valmass(x,y,3),label='q = 3')
plt.plot(x_y,get_def_window_valmass(x,y,8),label='q = 8')
plt.plot(x_y,get_def_window_valmass(x,y,10),label='q = 10')
plt.plot(x_y,get_def_window_valmass(x,y,30),label='q = 30')
plt.grid()
plt.legend()
plt.xlabel('a/b')
plt.ylabel('f(a,b)')
plt.show()