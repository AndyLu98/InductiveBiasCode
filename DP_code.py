import numpy as np

#DP algorithm for segmented regression. 
#Most credit to https://github.com/solohikertoo/segmented-least-squares/

#Calculate the regression error for the given segment. 
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


# Constuct fitted line from optimum solution (This function is finding the fit to plot) 
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


#This function is used to extract cooridnates of the end points of each intervals. 
'''
Input:
n --- number of data points
xarray ---- The x coordinates of the data points
yarray ---- The y coordinates of the data points
preresult --- The precomputed least square errors
opt ---- result from findopt function

Output:
x_cord -- the x coordinates of each of the segment.
y_cord -- the y coordinates of each of the segment.
segments -- an array containing each of the segment.
'''
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


# Segmented least squares with the coordinates of the endpoints for each interval. 

#This function is used to extract cooridnates of the end points of each intervals. 
'''
Input:
n --- number of data points
xarray ---- The x coordinates of the data points
yarray ---- The y coordinates of the data points
Cfactor ---- The penalty cost for DP

Output:
x_cord -- the x coordinates of each of the segment.
y_cord -- the y coordinates of each of the segment.
segments -- an array containing each of the segment.
'''

def segls(n, xarray, yarray, Cfactor):
    preresult = precompute(n, xarray, yarray)  # precompute ls coefs and errors
    opt = findopt(n, preresult, Cfactor)  # optimum solution
    #yfit = constructfit(n, xarray, yarray, preresult, opt)  # compute fit (for plotting)
    x_co, y_co, segs = constructInterval(n, xarray, yarray, preresult, opt)
    #return yfit
    return x_co, y_co, segs


# Calculate the MSE error of the answer. The num_data_point is optional and is for calculating the average MSE error. 
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


#Sample Usage: 
N = 256
MaxInput = 3
MinInput = -3
x_temp = np.arange(MinInput, MaxInput, (MaxInput - MinInput) / N)  #Data point x coordinates
y_temp = np.sin(2 * x_temp) #Data point y coordinates

#x_fit and y_fit can be used for plotting. Segments is used to compute the MSE of the result. 
x_fit, y_fit, segments = segls(N, x_temp, y_temp, c)  
mse = MSE_of_answer(segments, x_temp, y_temp, N)  


########This code is for finding the complexity of a given function defined by: 
#K(f, epsilon) = Piece_min(function, Data, epsilon)
# f -- function
# epsilon: training loss cutoff. 
# Piece_min: Minimum number of pieces by DP needed to achieve this epsilon.
# Actually not able to find the most exact value since the number of pieces is not uniformly distributed. 
# Find the best estimate. 
epsilon = 0.001 

def FindComplexityOfFunction(xarray, yarray, epsilon):

    loss = 10000000

    penalty = 10000

    penalty_lower = None

    penalty_upper = None

    while loss > epsilon:
        penalty_upper = penalty * 2.0
        x_fit, y_fit, segments = segls(N, x_temp, y_temp, penalty)
        penalty_lower = penalty
        penalty = penalty / 2.0  
        mse = MSE_of_answer(segments, x_temp, y_temp, N)
        number_of_pieces = len(segments)
        loss = mse

    penalty_old = penalty

    while (True):

        penalty = (penalty_upper + penalty_lower) / 2.0
        penalty_new = penalty

        if MSE_of_answer(segls(N, x_temp, y_temp, penalty)[2], x_temp, y_temp, N) < epsilon:
            penalty_lower = penalty 

        elif MSE_of_answer(segls(N, x_temp, y_temp, penalty)[2], x_temp, y_temp, N) > epsilon:
            penalty_upper = penalty

        print(penalty_old)
        print(penalty_new)

        if abs(penalty_new - penalty_old) < 10e-10:
            print(MSE_of_answer(segls(N, x_temp, y_temp, penalty)[2], x_temp, y_temp, N))
            break

        penalty_old = penalty_new

    return len(segls(N, x_temp, y_temp, penalty)[2])


print "Complexity of the Function with respect to epsilon:" + str(epsilon) + ": " + str(FindComplexityOfFunction(x_temp, y_temp, epsilon))
