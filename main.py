import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import random
import math
import time
import threading

from MLP import MultiLayerPerceptron
from MLP1 import MultiLayerPerceptron1
from point import pointD

def setAxes(ax):
	ax.set_xlim3d([-15, 15])
	ax.set_ylim3d([-15, 15])
	ax.set_zlim3d([-11, 11])
	ax.set_autoscale_on(False)

def generatePoint(radius, distance, width, label, neutral):
	# generate position on disk
	# https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
	r = width/2 * math.sqrt(random.random())
	theta = 2 * math.pi * random.random()
	x = r * math.cos(theta)
	y = r * math.sin(theta)
	
	if label == 'top':
		theta = math.pi * (random.random())
		
		disk_position = [x * math.cos(theta), y, x * math.sin(theta)]
		position_on_circumference = [radius * math.cos(theta), 0, radius * math.sin(theta)]
		central_displacement = [-radius/2, 0, distance/2]
		
		output_point = zip(disk_position, position_on_circumference, central_displacement)
		output_point = [sum(item) for item in output_point]
	
		if neutral:
			return pointD(output_point, -1)
		return pointD(output_point, 0)

	if label == 'bottom':
		theta = math.pi * (random.random()+1)
		
		disk_position = [x * math.cos(theta), y, x * math.sin(theta)]
		position_on_circumference = [radius * math.cos(theta), 0, radius * math.sin(theta)]
		central_displacement = [radius/2, 0, -distance/2]
		
		output_point = zip(disk_position, position_on_circumference, central_displacement)
		output_point = [sum(item) for item in output_point]
		
		if neutral:
			return pointD(output_point, -1)
		return pointD(output_point, 1)
	
def generateHalfMoon(amount, radius, distance, width, label, neutral):
	outputList = []
	i = 0
	while(i < amount):
		outputList.append(generatePoint(radius, distance, width, label, neutral))
		i += 1
		
	return outputList

def generateList(inputList):
	x = []
	y = []
	z = []
	colors = []
	
	i = 0
	while(i < len(inputList)):
		point = inputList[i]
		x.append(point.point[0])
		y.append(point.point[1])
		z.append(point.point[2])
		colors.append(point.getColor())
		i += 1
		
	return x,y,z,colors

def label_to_color(label):
	if label < 0.5:
		return 'red'
	elif label > 0.5:
		return 'blue'
	return 'grey'

def main():
	plt.figure

	plt.figure(figsize=(9,5))
	plt.figure = plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
	ax = plt.axes(projection='3d')
	plt.axis('off')
	setAxes(ax)

	# Generate Half Moons
	radius = 10
	distance = -5
	width = 5
	amount = 1000
	
	training_set = []
	training_set += generateHalfMoon(amount, radius, distance, width, 'top', False)
	training_set += generateHalfMoon(amount, radius, distance, width, 'bottom', False)
	
	x,y,z,colors = generateList(training_set)

	global scat
	scat = ax.scatter(x, y, z, color=colors)

	training_set_in = []
	training_set_target = []
	
	i = 0
	while(i < len(training_set)):
		point = training_set[i]
		training_set_in.append(point.point)
		training_set_target.append(point.label)
		i += 1

	nn = MultiLayerPerceptron1((0.3, 0.0, 20000), 3, 40, 1)
	nn.train(training_set_in, training_set_target)

	plt.pause(3)
	
	test_set = []
	test_set += generateHalfMoon(amount, radius, distance, width, 'top', True)
	test_set += generateHalfMoon(amount, radius, distance, width, 'bottom', True)

	random.shuffle(test_set)

	X,Y,Z,colors = generateList(test_set)
	
	scat.remove()
	scat = ax.scatter(X, Y, Z, color=colors)

	plt.pause(0.1)

	for i in range(len(test_set)):
		colors[i] = label_to_color(nn.activatePerceptron(test_set[i].point))

	def plane():
		x_list = np.linspace(-15,15,15)
		y_list = np.linspace(-10,10,10)
		x_list,y_list = np.meshgrid(x_list,y_list)
		z_list = np.array([[0]*len(x_list[0]) for i in range(len(x_list))])
		z_list = z_list.astype(np.float)
		rate = 0.5
		for i in range(len(z_list)):
			for j in range(len(z_list[i])):
				x = x_list[i][j]
				y = y_list[i][j]
				value = 0
				z = 0
				while value < 0.49 or value > 0.51:
					value = nn.activatePerceptron([x,y,z])[0][0]
					error = rate * (value - 0.5)
					z += error
					print(value)
				z_list[i][j] = z
		
		global scat
		scat.remove()
		scat = ax.scatter(X, Y, Z, color=colors)

		_colormap = 'terrain'
		ax.plot_surface(x_list, y_list, z_list, alpha=0.5, cmap=_colormap)

	p = threading.Thread(target=plane)
	p.start()

	plt.show()

if __name__ == "__main__":
  main()