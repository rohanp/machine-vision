#################################
#		Machine Vision			#
#		by Rohan Pandit			#
#################################

from __future__ import division

import numpy as np
import sys
import os
import subprocess
from math import atan2, pi

processes = set()

def main():
	#reading input
	infileName = sys.argv[1]+".ppm"
	f = open(infileName,'r').read().split() #list
	
	f.pop(0)
	width = int(f.pop(0))
	height = int(f.pop(0))
	maxColor = int(f.pop(0))

	img = np.array(f, dtype=np.int32)
	img = np.reshape(img, (height, width, 3))
	processes.add(subprocess.Popen("display %s"%infileName, shell=True))

	"""#img processing
	bw = grayscale(img)
	writeAndDisplayOutput(bw, 'bw')

	for i in range(6):
		bwBlurred = blur(bw)
	writeAndDisplayOutput(bwBlurred, 'bwBlurred')

	for i in range(6):
		coloredBlurred = blur(img)
	writeAndDisplayOutput(coloredBlurred, 'coloredBlurred')"""
	

	infileName = sys.argv[1]+"_bwBlurred.ppm"
	bwBlurredFile = open(infileName, 'r').read().split() 
	bwBlurredFile.pop(0)
	width = int(bwBlurredFile.pop(0))
	height = int(bwBlurredFile.pop(0))
	maxColor = int(bwBlurredFile.pop(0))

	blurred = np.array(bwBlurredFile, dtype=np.int32)
	blurred = np.reshape(blurred, (height, width, 3))
	bwBlurred = blurred[:,:,0:1]
	processes.add(subprocess.Popen("display %s"%infileName, shell=True))

	sobel = sobelMask(bwBlurred)
	imgAndEdges, edges = fatedges(blurred.copy(), sobel)
	writeAndDisplayOutput(imgAndEdges, 'fatEdges')

	imgAndEdges, edges = thinEdges(blurred.copy(), edges, sobel)
	writeAndDisplayOutput(imgAndEdges, 'thinEdges')



dirs = np.array( [(1,0), (1,1), (0,1), (-1,1)], dtype=int ) #directions

def circleDetection(edges, sobel):
	votes = np.zeros(edges.shape)

	for i in range(edges.shape[0]):
		for j in range(edges.shape[1]):
			if edges[i][j] == 1:
				for r in range(-edges.shape[0]/2, edges.shape[0]/2):
					direction = dirs[ sobel[i][j][0][1] ]
					center = np.add((i, j) + direction * r)




def canny(img, edges, sobel):
	pass

def thinEdges(img, edges, sobel):
	for i in range(sobel.shape[0]-1):
		for j in range(sobel.shape[1]-1):
			for c in range(sobel.shape[2]):
					if edges[i][j] == 1:
						theta = sobel[i][j][c][1]

						compare1 = sobel[ i + dirs[theta][0], j + dirs[theta][1], c, 0 ] 
						#compare1 = sobel[*np.add( (i,j), dirs[theta] ), c, 0]

						compare2 = sobel[ i - dirs[theta][0], j - dirs[theta][1], c, 0 ]
						if sobel[i][j][c][0] < compare1 or sobel[i][j][c][0] < compare2:
							edges[i][j] = 0
						else:
							img[i][j] = [255, 0 , 0]

	return img, edges



def fatedges(img, sobel):
	edges = np.zeros(img.shape[0:2])
	for i in range(sobel.shape[0]-1):
		for j in range(sobel.shape[1]-1):
			for c in range(sobel.shape[2]):
					if sobel[i][j][c][2] == 1:
						img[i][j] = [255, 0, 0]
						edges[i][j] = 1
	
	return img, edges


def sobelMask(img):
	sobel = np.zeros((img.shape[0], img.shape[1], img.shape[2], 5))
	sx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	sy = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

	for i in range(1, img.shape[0]-1):
		for j in range(1, img.shape[1]-1):
			for c in range(img.shape[2]):
				gx = np.tensordot(img[i-1:i+2, j-1:j+2, c], sx)
				gy = np.tensordot(img[i-1:i+2, j-1:j+2, c], sy)
				M = np.sqrt(gx*gx + gy*gy)
				D = theta(gx, gy)
				if M > 120:
					edge = 1
				else:
					edge = 0

				sobel[i][j][c] = M, D, edge, 0, 0
	
	return sobel

def theta(x, y):
	return int((abs(atan2(y,x))/3.142)*4) #scales from 0-pi to 0-3

def blur(img):
	blurred = np.zeros(img.shape) 
	smoothingMask = np.array([[1,2,1], [2,4,2], [1,2,1]])

	for j in range(1, img.shape[1]-1):
		for i in range(1, img.shape[0]-1):
			for c in range(img.shape[2]):
				blurred[i][j][c] = int(np.tensordot(img[i-1:i+2, j-1:j+2, c], smoothingMask)/16)

	return blurred


def grayscale(img):
	height = img.shape[0]
	width = img.shape[1]	
	bw = np.zeros((height,width,1))

	for i in range(height):
		for j in range(width):
			bw[i][j] = int(img[i][j][0]*.2 + img[i][j][1]*.7 + img[i][j][2]*.1) #grayscale

	return bw

def toPPM(array):
	img = np.zeros(array.shape[0], array.shape[1], 3)

	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i][j] == 1:
				img[i][j] = [255, 0, 0]

	return img

def writeAndDisplayOutput(img, ext): #ext is what filename should end with
	outfileName = sys.argv[1] + "_" + ext + ".ppm"
	height = img.shape[0]
	width = img.shape[1]
	colors = 3		
	maxColor = 255

	outfile = open(outfileName, 'w')
	outfile.write('P3\n')
	outfile.write(str(width) + " " + str(height) + '\n')
	outfile.write('%s\n'%maxColor)

	for i in range(height):
		for j in range(width):
			for k in range(colors):
				try:
					outfile.write('%d '%img[i][j][k])
				except IndexError:
					outfile.write('%d '%img[i][j][0])

	processes.add(subprocess.Popen("display %s"%outfileName, shell=True))

if __name__ == "__main__":
	main()	