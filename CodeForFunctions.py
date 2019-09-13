
import torch
import matplotlib as mpl
from matplotlib import cm
from matplotlib.pyplot import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from scipy import signal
import scipy

np.random.seed(0)
random.seed(42)


N, D_in, H, D_out = 256, 1, 10000, 1
NumIters = 5000
PlotFreq = 200
PrintFreq = 100

MinInput = -3.
MaxInput = +3.
NoiseStdDev = 0.

a = np.random.uniform(low = MinInput, high = MaxInput, size = (N, D_in))
x = torch.Tensor(a)

noise = NoiseStdDev * torch.randn(N, D_out)


###########The Functions Implementation###########

y = torch.sin(2 * x) + noise #sine

y = x * torch.atan(x / 2.0) + noise  #arctan

y = torch.cos(2 * x) - 1.0 + noise #cosine activation

y = torch.Tensor(signal.sawtooth(2 * np.pi * x, 0.5)) #Piece-wise linear function triangular-wave

y = 0.5 * x * x + 0.25 * x * x * x - 0.5 * x  #cubic 

y = 0.5 * x * x                           #quadratic 

#These two lines should be run together.
y = x * torch.atan(x / 2.0) 
y = 2.5*(y-1.1) + 2.1*(y+.35) - (y+.81) + 1.5*(y+.26) - 3.1*(y-.23)  #self composition of arctan


y = torch.exp(0.5 * x) + noise                     #exponential 


#This is the Takagi function based on this paper: https://arxiv.org/pdf/1110.1691.pdf
x = np.random.uniform(low = 0, high = 1, size = (N, D_in))

y = np.zeros(shape = (N, D_in))
for i in range(1024):
     y += ((1.0 / (2.0**i)) * np.abs((np.rint(1.0 * (2.0**i) * x)) - (1.0 * (2.0**i) * x)))

x = torch.Tensor(x)
y = torch.Tensor(y)
