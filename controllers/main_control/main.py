import sys
from SPAIC import spaic
import matplotlib.pyplot as plt


a = [1,2,3,4,5]
b = [1,1,1,1,1]
c = [2,2,2,2,2]

plt.subplot(2,2,1)
plt.plot(a,b)

plt.subplot(2,2,2)
plt.plot(a,c)

plt.show()