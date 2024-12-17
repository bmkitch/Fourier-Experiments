import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy as sp
import tqdm

"""
The goal of this file is similar to 1dFourier except with more of a focus on the transorm itself. I left this file with a display of
how a double Fourier transform is a mirror image of the inverse Fourier transform.
"""


def field(x, y, z):
    total = 0
    for k in range(1, 100):
        total += (np.sin(z*x/k) + np.cos(z*y/k))**2
    return total


#creating the grid of values based on the functions above
X = np.linspace(-5.0, 5.0, 1000)
Y = np.linspace(-5.0, 5.0, 1000)
z = 10

grid = np.array([field(i, j, z) for j in tqdm.tqdm(Y) for i in X])

#reshaping grid data into square before 2d transforms
grid = grid.reshape(1000, 1000)

#finding the three transforms
transform = sp.fft.fft2(grid)
transform2 = sp.fft.fft2(transform)
transform3 = sp.fft.fft2(transform2)

#Getting the real components
real = np.real(transform)
real2 = np.real(transform2)
real3 = np.real(transform3)

#plotting
f, images = plt.subplots(2, 2)
images[0, 0].imshow(grid, interpolation='none')
images[0, 0].title.set_text("Original Grid")
images[1, 0].imshow(np.abs(real[:-900, :-900]), interpolation='none', norm=LogNorm())
images[1, 0].title.set_text("One Transform")
images[0, 1].imshow(real2, interpolation='none')
images[0, 1].title.set_text("Two Transforms")
images[1, 1].imshow(np.abs(real3[:-900, :-900]), interpolation='none', norm=LogNorm())
images[1, 1].title.set_text("Three Transforms")
plt.show()