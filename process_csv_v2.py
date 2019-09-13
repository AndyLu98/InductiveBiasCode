import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import sys


#To use, simply replace the link to the csv with your own path to the file. It will automatically plot the result and 
# error bar as long as the  csv row is in the format: Number of hidden units/Pieces, Final training loss, Tag. 

mpl.style.use('seaborn')
cmap = mpl.cm.get_cmap('plasma')


#Load and plot result for regular SGD
dictionary_for_GD_value = {}

with open('./csv_for_comparison/quadratic_-3_to_3_no_more_fixed_seed.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[2] == "SGD":
			if int(row[0]) not in dictionary_for_GD_value:
				dictionary_for_GD_value[int(row[0])] = [float(row[1]), float(row[1]), float(row[1]), 1]
			else:
				dictionary_for_GD_value[int(row[0])] = [float(row[1]) + dictionary_for_GD_value[int(row[0])][0], 
				min(dictionary_for_GD_value[int(row[0])][1], float(row[1])), max(dictionary_for_GD_value[int(row[0])][2], float(row[1])), 
				dictionary_for_GD_value[int(row[0])][3] + 1]


for key in dictionary_for_GD_value:
	dictionary_for_GD_value[key][0] = dictionary_for_GD_value[key][0] / dictionary_for_GD_value[key][3]

x_val = []
y_val = []
yerror = []
for key in sorted(dictionary_for_GD_value.keys()):
	x_val.append(key)
	y_val.append(dictionary_for_GD_value[key][0])
	yerror.append(abs(dictionary_for_GD_value[key][0] - dictionary_for_GD_value[key][1]))

for key in sorted(dictionary_for_GD_value.keys()):
	yerror.append(abs(dictionary_for_GD_value[key][0] - dictionary_for_GD_value[key][2]))

print (np.shape(yerror))
yerror = np.asarray(yerror)
yerror = yerror.reshape(2,8)
print yerror



dictionary_for_ADAM_value = {}
with open('./csv_for_comparison/quadratic_-3_to_3_no_more_fixed_seed.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[2] == "ADAM":
			if int(row[0]) not in dictionary_for_ADAM_value:
				dictionary_for_ADAM_value[int(row[0])] = [float(row[1]), float(row[1]), float(row[1]), 1]
			else:
				dictionary_for_ADAM_value[int(row[0])] = [float(row[1]) + dictionary_for_ADAM_value[int(row[0])][0], 
				min(dictionary_for_ADAM_value[int(row[0])][1], float(row[1])), max(dictionary_for_ADAM_value[int(row[0])][2], float(row[1])), 
				dictionary_for_ADAM_value[int(row[0])][3] + 1]


for key in dictionary_for_ADAM_value:
	dictionary_for_ADAM_value[key][0] = dictionary_for_ADAM_value[key][0] / dictionary_for_ADAM_value[key][3]

x_val_ADAM = []
y_val_ADAM = []
yerror_ADAM = []
for key in sorted(dictionary_for_ADAM_value.keys()):
	x_val_ADAM.append(key)
	y_val_ADAM.append(dictionary_for_ADAM_value[key][0])
	yerror_ADAM.append(abs(dictionary_for_ADAM_value[key][0] - dictionary_for_ADAM_value[key][1]))

for key in sorted(dictionary_for_ADAM_value.keys()):
	yerror_ADAM.append(abs(dictionary_for_ADAM_value[key][0] - dictionary_for_ADAM_value[key][2]))

print (np.shape(yerror_ADAM))
yerror_ADAM = np.asarray(yerror_ADAM)
yerror_ADAM = yerror_ADAM.reshape(2,8)
print yerror_ADAM


dp_value_pair = {}
with open('./csv_for_comparison/quadratic_-3_to_3_no_more_fixed_seed.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
    	if row[2] == "DP":
	    	if row[0] not in dp_value_pair:
	    		dp_value_pair[int(row[0])] = row[1].replace("E","e").replace("[","").replace("]","")
	    	else:
	    		dp_value_pair[int(row[0])] = row[1].replace("E","e").replace("[","").replace("]","")


dp_x_val = []
dp_y_val = []

for key in sorted(dp_value_pair.keys()):
	dp_x_val.append(key)
	dp_y_val.append(float(dp_value_pair[key]))



greedy_result = {}
t = 0
z = 0
with open('./csv_for_comparison/quadratic_-3_to_3_no_more_fixed_seed.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')

	for row in csv_reader:
		z += 1
		if row[2] == "GREEDY":
		   t += 1
		   if int(row[0]) not in greedy_result:
				greedy_result[int(row[0])] = [float(row[1]), float(row[1]), float(row[1]), 1]
		   else:
				greedy_result[int(row[0])] = [float(row[1]) + greedy_result[int(row[0])][0], 
				min(greedy_result[int(row[0])][1], float(row[1])), max(greedy_result[int(row[0])][2], float(row[1])), 
				greedy_result[int(row[0])][3] + 1]


greedy_x_val = []
greedy_y_val = []
greedy_error = []


print(greedy_result)
for key in greedy_result:
	greedy_result[key][0] = greedy_result[key][0] / greedy_result[key][3]

for key in sorted(greedy_result.keys()):
	greedy_x_val.append(key)
	greedy_y_val.append(greedy_result[key][0])
	greedy_error.append(abs(greedy_result[key][0] - greedy_result[key][1]))

for key in sorted(greedy_result.keys()):
	greedy_error.append(abs(greedy_result[key][0] - greedy_result[key][2]))

print(greedy_error)
greedy_error = np.asarray(greedy_error)
greedy_error = greedy_error.reshape(2,52)




dictionary_for_GD_batch_value = {}

with open('./csv_for_comparison/quadratic_-3_to_3_no_more_fixed_seed.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[2] == "BATCHNORM SGD":
			if int(row[0]) not in dictionary_for_GD_batch_value:
				dictionary_for_GD_batch_value[int(row[0])] = [float(row[1]), float(row[1]), float(row[1]), 1]
			else:
				dictionary_for_GD_batch_value[int(row[0])] = [float(row[1]) + dictionary_for_GD_batch_value[int(row[0])][0], 
				min(dictionary_for_GD_batch_value[int(row[0])][1], float(row[1])), max(dictionary_for_GD_batch_value[int(row[0])][2], float(row[1])), 
				dictionary_for_GD_batch_value[int(row[0])][3] + 1]


for key in dictionary_for_GD_batch_value:
	dictionary_for_GD_batch_value[key][0] = dictionary_for_GD_batch_value[key][0] / dictionary_for_GD_batch_value[key][3]

x_val_batch = []
y_val_batch = []
yerror_batch = []
for key in sorted(dictionary_for_GD_batch_value.keys()):
	x_val_batch.append(key)
	y_val_batch.append(dictionary_for_GD_batch_value[key][0])
	yerror_batch.append(abs(dictionary_for_GD_batch_value[key][0] - dictionary_for_GD_batch_value[key][1]))

for key in sorted(dictionary_for_GD_batch_value.keys()):
	yerror_batch.append(abs(dictionary_for_GD_batch_value[key][0] - dictionary_for_GD_batch_value[key][2]))

print (np.shape(yerror_batch))
yerror_batch = np.asarray(yerror_batch)
yerror_batch = yerror_batch.reshape(2,7)
print yerror_batch




fig = plt.figure()
ax = plt.axes()
ax.set_yscale("log")
ax.set_xscale("log")
plt.errorbar(x_val, y_val, yerr = yerror, fmt = '-o',ecolor='black',capthick=2, label = "Gradient Descent")
plt.errorbar(x_val_ADAM, y_val_ADAM, yerr = yerror_ADAM, fmt = '-o', color = "cyan", ecolor='green',capthick=2, label = "Gradient Descent with ADAM")
plt.plot(dp_x_val, dp_y_val, "-o", color = "red", label = "DP algorithm")
plt.errorbar(greedy_x_val, greedy_y_val, yerr = greedy_error, fmt = "-o", color = "green", label = "Greedy Algorithm")
plt.errorbar(x_val_batch, y_val_batch, yerr = yerror_batch, fmt = "-o", color = "orange", label = "SGD with Batchnorm Algorithm")
ax.set_xlabel("Number of Pieces")
ax.set_ylabel("Training Loss")
plt.legend(loc='upper right')
plt.title("Fitting Quadratic function with different algorithms")

plt.show()