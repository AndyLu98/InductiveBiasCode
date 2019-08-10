import julia
from julia import Julia
import numpy as np
from julia import Base
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import csv
import random

np.random.seed(0)
random.seed(42)
#To run the code, first needs to download julia 1.1 and then set its ENVIRONMENTAL PATH. 

#These needs to be done if want to run the code and plot the result. 
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/4.1.4/bin/ffmpeg'  #Change the path for your own computer
mpl.style.use('seaborn')
cmap = mpl.cm.get_cmap('plasma')
mywriter = animation.FFMpegWriter(fps=100, metadata=dict(artist='Me'), bitrate=18000)

julia.install("/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia")  #Change the path for your own computer
j = julia.Julia()
j.include("./fast-segmented-regression-master/src/linear_merging.jl") #Change the path for your own computer

########THEIR CODE FOR GENERATING DATA##############
# endpoint_values = Base.rand([1,2,3,4,5,6,7,8,9,10], 6)
# n = 500
# k = len(endpoint_values) - 1
# sigma = 1.0
# y, ystar, X = j.generate_equal_size_linear_data(endpoint_values, n, sigma)
# print(X)
# print(endpoint_values)

# print(np.shape(X))
# print(np.shape(ystar))
# print(np.shape(y))
####################################################

def mse(breakpoints, x, y, number_data_point):
	MSE = 0.0
	for segment in breakpoints:

		#JULIA is 1 indexed, but python is 0 indexed 
		leftPoint = segment.left_index - 1
		rightPoint = segment.right_index - 1

		leftPointXVal = x[leftPoint][1]
		leftPointYVal = y[leftPoint] 

		rightPointXVal = x[rightPoint][1]
		rightPointYVal = y[rightPoint]

		# Slope of the segment
		m = (rightPointYVal - leftPointYVal) / (1.0 * (rightPointXVal - leftPointXVal))

		# Y intercept of the segment
		b = leftPointYVal

		for point in range(leftPoint, rightPoint + 1):
			MSE += (y[point] - (m * (x[point][1] - x[leftPoint][1]) + b))**2

	return MSE #/ number_data_point


#Sample Usage: 
result = {}

number_data_point = 256

#First create an empty array
X = np.empty(shape = (number_data_point, 2))

A = np.random.uniform(low= -5.0, high=5.0, size = (number_data_point, 1))

X.reshape(number_data_point, 2)

X[:,0] = 1.0 #This is what has to be done according to the original Julia Code. 
X[:,1] = np.sort(A[:,0], axis = 0)

y = np.sin(2.0 * X[:,1])

sigma = 0.0

yhat_merging_partition, past_iterations = j.fit_linear_merging(X, y, sigma, 2 * k, k, initial_merging_size=2)

print("number of intervals:" + str(len(yhat_merging_partition)))

MSE = mse(yhat_merging_partition, X, y, number_data_point)

print(MSE)



#This part of the code is used for generating evolution of interval as a function of time step. 

yhat_merging_partition, past_iterations = j.fit_linear_merging(X, y, sigma, 2 * k, k, initial_merging_size=2)
print(yhat_merging_partition)
print("number of intervals:" + str(len(yhat_merging_partition)))

fig = plt.figure()
plt.plot(y)

ims = []

print("num time step: " + str(len(past_iterations)))

for i in range(len(past_iterations)):
	x_loc = []
	color = cmap(i / len(past_iterations) * 1.0)
	for interval in past_iterations[i]:
		x_loc.append(interval.left_index)
		x_loc.append(interval.right_index)

	y_loc = []
	for item in x_loc:
		y_loc.append(y[item - 1])

	curPlot, = plt.plot(x_loc,y_loc,'ro',
             linewidth=1.0, color= color,animated=True)
	timeStep = plt.text(100, 0.8, 'time_step = ' + str(i))

	ims.append([curPlot,timeStep])


intervals = []
x_loc = []
for interval in yhat_merging_partition:
	intervals.append(interval)
	x_loc.append(interval.left_index)
	x_loc.append(interval.right_index)

y_loc = []
for item in x_loc:
	y_loc.append(y[item - 1])


d, = plt.plot(x_loc, y_loc,"-o")
timeStep = plt.text(100, 0.8, 'time_step = ' + str(len(past_iterations) + 1))
ims.append([d,timeStep])

fname = './Test1/Figures/result_from_july_15/fast-segmented-regression-algorithm-break-points-evolution_10' #Change the pathname for your own computer. 

ani = animation.ArtistAnimation(fig,ims,interval=50,blit = True,repeat_delay=1000)
ani.save(fname+'.mp4', writer = mywriter)
print(fname)

plt.show()