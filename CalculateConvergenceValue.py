# -*- coding: utf-8 -*-
# import numpy as np
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

mpl.style.use('seaborn')
cmap = matplotlib.cm.get_cmap('winter')

torch.set_printoptions(precision=10)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 256, 1, 10000, 1
NumIters = 500000
PlotFreq = 200
PrintFreq = 100

MinInput = -3.
MaxInput = +3.
NoiseStdDev = 0.

print(H)
# Create random Tensors to hold inputs and outputs

a = np.random.uniform(low = MinInput, high = MaxInput, size = (N, D_in))
x = torch.Tensor(a)

noise = NoiseStdDev * torch.randn(N, D_out)

y = torch.sin(2 * x)
# y = x * torch.atan(x / 2.0) + noise  #arctan
# y = torch.cos(2 * x) - 1.0 + noise  #cosine activation

# y = torch.Tensor(signal.sawtooth(2 * np.pi * x, 0.5)) #Piece-wise linear function triangular-wave

#y = 0.5 * x * x + 0.25 * x * x * x - 0.5 * x  #cubic - dp tested
#y = 0.5 * x * x                           #quadratic - dp tested

# y = 2.5*(y-1.1) + 2.1*(y+.35) - (y+.81) + 1.5*(y+.26) - 3.1*(y-.23)  #self composition of arctan
# y = torch.exp(0.5 * x) + noise                      #exponential -dp tested


#This is the Takagi function based on this paper: https://arxiv.org/pdf/1110.1691.pdf
# x = np.random.uniform(low = 0, high = 1, size = (N, D_in))
#
# y = np.zeros(shape = (N, D_in))
# for i in range(1024):
#     y += ((1.0 / (2.0**i)) * np.abs((np.rint(1.0 * (2.0**i) * x)) - (1.0 * (2.0**i) * x)))
# x = torch.Tensor(x)
# y = torch.Tensor(y)


learning_rate = 2e-6

#####SEGMENTED_REGRESSSION_PART:

#Most credit to https://github.com/solohikertoo/segmented-least-squares/

# x = np.array(a)


# y = 0.5 * x * x + 0.25 * x * x * x - 0.5 * x  #cubic
# y = 0.5 * x * x                           #quadratic 
# y = np.exp(0.5 * x)                      #exponential 
# y = np.array(signal.sawtooth(2 * np.pi * x, 0.5)) #Piece-wise linear function triangular-wave
#
# y = x * np.arctan(x / 2.0)  #arctan
# y = 2.5*(y-1.1) + 2.1*(y+.35) - (y+.81) + 1.5*(y+.26) - 3.1*(y-.23)  #self composition of arctan
#
#
# Takagi function
# x = np.random.uniform(low = 0, high = 1, size = (N, D_in))
# y = np.zeros(shape = (N, D_in))
# for i in range(1024):
#     y += ((1.0 / (2.0**i)) * np.abs((np.rint(1.0 * (2.0**i) * x)) - (1.0 * (2.0**i) * x)))
#
# y = np.cos(2 * x) - 1.0 #cosine activation
# y = np.sin(2 * x) # sine function


def lscoefs(xarr, yarr):
    n = len(xarr)
    if (n == 1):
        return (0.0, yarr[0], 0.0)
    sx = sum(xarr)
    sy = sum(yarr)
    sx2 = sum([x * x for x in xarr])
    sxy = 0
    for i in range(n):
        sxy = sxy + xarr[i] * yarr[i]
    a = (n * sxy - sx * sy) / (n * sx2 - sx * sx)
    b = (sy - a * sx) / n
    e = 0
    for i in range(n):
        e = e + (yarr[i] - a * xarr[i] - b) ** 2
    return (a, b, e)


# precompute all least squares coefs for all pairs of points i <= j
# result is a list of lists, one per j, each entry in sublist is a
# tuple for an i
# each tuple is the ls coefs a, b and the error, for that segment (i,j)
def precompute(n, xarray, yarray):
    result = []
    for j in range(n):
        print(j)
        jres = []
        for i in range(0, j + 1):
            a, b, e = lscoefs(xarray[i:j + 1], yarray[i:j + 1])
            jres = jres + [(a, b, e)]
        result = result + [jres]
    return (result)

# dynamic programming solution using precomputed results
def findopt(n, preresult, C):
    optresult = []
    for j in range(0, n):
        beste = 9e999
        besti = -1
        jpre = preresult[j]  # list of tuples (a,b,c) for i=0,1,...,j
        # for each possible start i, up to j
        for i in range(0, j + 1):
            # get the error assuming using opt to to i-1, and new fit for i..j,
            # with penalty per segment, C
            if (i > 0):
                e = jpre[i][2] + optresult[i - 1][0] + C
            else:
                e = jpre[i][2] + C
            # find i with smallest error
            if (e < beste):
                beste = e
                besti = i
        # create opt entry for j, consisting of min error and list of of (i,j)
        # this is this a list of (i,j) segments for this j, consisting of
        # the current best (i,j) appended to list from opt[besti-1]
        if (besti > 0):
            optresult = optresult + [[beste, optresult[besti - 1][1] + [(besti, j)]]]
        else:
            optresult = optresult + [[beste, [(besti, j)]]]
    return optresult[n - 1]


# constuct fitted line from optimum solution
def constructfit(n, xarray, yarray, preresult, opt):
    yfit = []
    optintervals = opt[1]  # list of tuples (opt has error and list of tuples)
    print(optintervals)
    print("number of optimal intervals:" + str(len(optintervals)))
    # for each segment
    for interval in optintervals:
        i = interval[0]  # get the segment
        j = interval[1]
        a = preresult[j][i][0]  # get the ls coeffs for the segment
        b = preresult[j][i][1]
        xarr = xarray[i:j + 1]
        yfit = yfit + [a * x + b for x in xarr]  # build up fit
    return yfit


def constructInterval(n, xarray, yarray, preresult, opt):
    x_cord = []
    y_cord = []
    segments = []
    optintervals = opt[1]  # list of tuples (opt has error and list of tuples)
    print(optintervals)
    print("number of optimal intervals:" + str(len(optintervals)))
    # for each segment
    for interval in optintervals:
        i = interval[0]  # get the segment
        j = interval[1]
        x_cord.append(xarray[i])
        y_cord.append(yarray[i])
        x_cord.append(xarray[j])
        y_cord.append(yarray[j])
        segments.append([i,j])
    return x_cord, y_cord, segments


# segmented least squares
def segls(n, xarray, yarray, Cfactor):
    preresult = precompute(n, xarray, yarray)  # precompute ls coefs and errors
    opt = findopt(n, preresult, Cfactor)  # optimum solution
    #yfit = constructfit(n, xarray, yarray, preresult, opt)  # compute fit
    x_co, y_co, segs = constructInterval(n, xarray, yarray, preresult, opt)
    #return yfit
    return x_co, y_co, segs


def segls_updated(n, xarray, yarray, Cfactor, precomputed_result):
    opt = findopt(n, precomputed_result, Cfactor)
    x_co, y_co, segs = constructInterval(n, xarray, yarray, precomputed_result, opt)
    # return yfit
    return x_co, y_co, segs


def MSE_of_answer(breakpoints, x, y, num_data_point):
    MSE = 0.0
    for segments in breakpoints:
        leftPoint = segments[0]
        rightPoint = segments[1]

        if x[rightPoint] == x[leftPoint]:
            slope = 0
        else:
            slope = (y[rightPoint] - y[leftPoint]) / (1.0 * (x[rightPoint] - x[leftPoint]))

        b = y[leftPoint]

        for point in range(leftPoint, rightPoint + 1):
            MSE += (y[point] - (slope * (x[point] - x[leftPoint]) + b))**2

    return MSE #/ num_data_point


result_for_dp = {}
C = []

j = 1.0

for _ in range(18):

    k = 1
    for i in range(8):
        C.append(j/k)
        k = k * 10.0
    j += 0.5


"""
This part of the code collects the result for DP algorithm. Collect the training loss corresponding to the number of pieces. 
"""
# PRECOMPUTED = precompute(N, x, y)
# for c in C:
#     x_fit, y_fit, segments = segls_updated(N, x, y, c, PRECOMPUTED)
#     mse = MSE_of_answer(segments, x, y, N)
#     if len(segments) not in result_for_dp:
#         result_for_dp[len(segments)] = mse
#         with open('./csv_for_comparison/cosine_SGD_loss_no_more_fixed_seed.csv', mode='a') as output_file:
#             output_file_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             output_file_writer.writerow([len(segments), mse, c, "DP"])
#     else:
#         print(".")
        #result_for_dp[len(segments)].append(mse)

#sys.exit()
##############



#######This part lets the neural network runs until convergence and then record the final training loss######

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(H),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

final_loss = None
threshold = 1e-10

last_loss = 10000000
current_loss = 99999
t = 0

while abs(last_loss - current_loss) > threshold:
    last_loss = current_loss
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    loss_sc = loss.item() ** 0.5
    current_loss = loss.item()
    final_loss = current_loss

    # Print loss
    if t % PrintFreq == 0:
        print(t, loss.item(), loss_sc)


    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Update parameters according to chosen optimizer
    optimizer.step()

    t += 1

print(H)
print("BATCHNORM")

"""
Write down the result in form of: number of neurons, final training loss, tag that indicates what training algorithm is used (e.g. batchnorm, adam, etc)
"""
with open('./csv_for_comparison/sine2x_SGD+BATCHNORM.csv', mode='a') as output_file:
    output_file_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_file_writer.writerow([str(H),str(final_loss), 'BATCHNORM SGD'])

print(final_loss)
print(t)

sys.exit()
