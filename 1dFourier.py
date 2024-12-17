import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

"""
The goal of this script was to become familiar with fourier transforms and to find functions that contain superoscillations.
After lots of experimenting and some research, I was able to use fourier to find multiple functions that appeared to have local
superoscillations. I hope this experience increases my ability to process data and find patterns.
"""

#first test function
def bellCurve(x, o):
    return (1/(o*np.sqrt(2*np.pi))) * np.power(np.e, -0.5 * (x / o) * (x / o))

#second test function
def wave(x):
    return np.sin(25*np.pi*x)

#Function that could have localized superoscillation
def powerWave(x):
    return np.power(np.sin(1*np.pi*x), 10)

#Another function with potential superoscillation
def ipowerWave(x):
    a = 2
    p = 30
    w = np.cos(1*np.pi*x) + a*np.sin(1*np.pi*x)*(1j)
    return np.power(w, p) / a**p

#interesting but probably not. (has strange implications for infinite frequencies. couldn't get there with this resolution)
def hyperFreq(x):
    l = 10
    return np.sin((100 + 100*l / (l+x*x)) * x)

#interesting but no superoscillation
def infFreq(x):
    return np.sin(1000/(x*x))

def powerWaveLoop(x):
    total = 0

    for a in range(24):
        total = total + np.power(np.sin(1*np.pi*x + 0.08400*np.pi*a), 1001)

    return total

def homework(x):
    return np.power(np.cos(x), 4)

N = 10000 #number of points   
T = 0.001 #spacing between points
X = np.linspace(-N*T, N*T, N)

o = 1000 #only for bell curve and sin2 functions

data = ipowerWave(X)

transform = sp.fft.fft(data)
freq = sp.fft.fftfreq(N, T)[:N//2] #[0:N//2] ensures its the same as the tranform when passed into the plot function


real = np.real(transform)


#transform[0:1000] = 0
#transform[100:] = 0

#transform[:] = 0
#transform[20000] = 1

back = sp.ifft(transform)

#plt.plot(freq, np.abs(transform[0:N//2]))
#plt.plot(data)
plt.plot(back)
plt.show()