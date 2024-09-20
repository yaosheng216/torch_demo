import matplotlib.pyplot as plt
import numpy as np

data = 2*np.random.rand(10000, 2) - 1
print(data)
x = data[:, 0]
y = data[:, 1]
idx = x**2 + y**2 < 1
hole = x**2 + y**2 < 0.25
idx = np.logical_and(idx, ~hole)
plt.plot(x[idx], y[idx], 'go', markersize=1)
plt.show()

p = np.random.rand(10000)
np.set_printoptions(edgeitems=5000, suppress=True)
plt.hist(p, bins=20, color='g', edgecolor='k')
plt.show()

n = 10000
times = 100
z = np.zeros(n)
for i in range(times):
    z += np.random.rand(n)
z /= times
plt.hist(z, bins=20, color='m', edgecolor='k')
plt.show()

d = np.random.rand(2, 3)
print(d)

x = np.arange(0, 10, 1)
y = x**x
plt.plot(x, y, 'r-', linewidth=3)
plt.show()
