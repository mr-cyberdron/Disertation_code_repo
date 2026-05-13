import matplotlib.pyplot as plt
import numpy as np
import wfdb
sigs, fields = wfdb.rdsamp('./11010_hr', channels=[0, 5, 7], sampfrom=0, sampto=5000)
# Note that you'll have to omit the filename extension; wfdb reads all three files at once

fig = plt.figure(figsize=(20, 6), dpi=144)
ax = fig.add_subplot(111, projection='3d')
ax.grid()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.scatter(sigs[:,0],sigs[:,1],sigs[:,2],color='r',marker='.',alpha=0.5)
plt.show()
