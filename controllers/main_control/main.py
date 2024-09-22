import sys
from SPAIC import spaic
import matplotlib.pyplot as plt
import time

a = [1,2,3,4,5]
b = [1,1,1,1,1]
c = [9,2,3,2,6]


plt.ion()
plt.figure(1)
plt.plot(a,b)
plt.show()
for _ in range(3):
    time.sleep(1)
    pass
plt.figure(2)
plt.plot(a,c)
plt.ioff()
plt.show()