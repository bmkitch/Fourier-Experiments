import numpy as np
from matplotlib import pyplot as plt

#Fun quick script to show the energy of a nonlinear wave

def wave(x):
    return np.power(0.5*np.cos(1*np.pi*x) + 1*np.sin(1*np.pi*x)*(1j), 10)

X = np.linspace(-20, 20, 10000)
Y = wave(X)

energy = np.conjugate(Y) * Y

plt.plot(Y)
plt.plot(energy)
plt.show()