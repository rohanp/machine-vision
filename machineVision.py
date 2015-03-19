#################################
#		Machine Vision			#
#		by Rohan Pandit			#
#################################

from __future__ import division

import numpy as np
from numpy import sqrt, sin, cos, tan, pi
import sys
import os
import subprocess
from math import atan2

processes = set()

def main():
	#reading input
	os.chdir("data")
	infileName = sys.argv[1]+".ppm"
	f = open(infileName,'r').read().split() #list
	
	f.pop(0)
	width = int(f.pop(0))
	height = int(f.pop(0))
	maxColor = int(f.pop(0))

	img = np.array(f, dtype=np.int32)
	img = np.reshape(img, (height, width, 3))
	#processes.add(subprocess.Popen("display %s"%infileName, shell=True))

	#img processing
	bw = grayscale(img)
	writeAndDisplayOutput(bw, 'bw')

	for i in range(1):
		bwBlurred = blur(bw)
	writeAndDisplayOutput(bwBlurred, 'bwBlurred')

	#for i in range(1):
	#	coloredBlurred = blur(img)
	#writeAndDisplayOutput(coloredBlurred, 'coloredBlurred')

	blurred = bwBlurred[:,:,0:1]
	sobel = sobelMask(blurred)
	np.save(sys.argv[1] + "_sobel.npy", sobel)
	imgAndEdges, edges = fatedges(img, sobel)
	writeAndDisplayOutput(imgAndEdges, 'fatEdges')

	imgAndEdges, edges = thinEdges(img, edges, sobel)
	writeAndDisplayOutput(imgAndEdges, 'thinEdges')
	np.save(sys.argv[1] + "_edges.npy", edges)

	sobel = np.load(sys.argv[1] + "_sobel.npy")
	edges = np.load(sys.argv[1] + "_edges.npy") 
	sobel = sobel[1:-1, 1:-1,:,:]
	edges = edges[1:-1, 1:-1]
	colorEdges = edges[:,:,np.newaxis] * np.array([255, 0, 0])
	writeAndDisplayOutput(colorEdges, "edges")

	center, radius = circleDetection(edges, sobel)
	print("center, radius:", center, radius)
	imgCircle = drawCircle(colorEdges, center, radius)
	writeAndDisplayOutput(imgCircle, "circleDetection")


def lineDetection(edges, sobel):
	pass

def drawCircle(img, center, r):
	for t in frange(0, 2*pi, 0.001):
		x = int( r * cos(t))
		y = int( r * sin(t))
		img[ center[0] + y, center[1] + x ] = [0, 0, 255]
	return img

def circleDetection(edges, sobel):
	votes = np.zeros( (edges.shape[0], edges.shape[1], int(edges.shape[0]/2)) ) #x, y, radius

	for i in range( votes.shape[0] ):
		for j in range( votes.shape[1] ):
			if edges[i][j]:
				for r in range( 1, votes.shape[2] ) :
					theta = sobel[i][j][0][1]
					centers = np.array([ ( i+r*sin(theta), j+r*cos(theta) ), ( i-r*sin(theta), j-r*cos(theta) ) ], dtype = int)
					for center in centers:
						if 0 < center[0] < votes.shape[0] and 0 < center[1] < votes.shape[1]:
							votes[center[0], center[1], r] += 1

	thinVotes = np.copy( votes )
	"""for i in range( votes.shape[0] ):
		for j in range( votes.shape[1] ):
			for r in range(1, votes.shape[2]-1 ):
				if votes[i][j][r-1] < votes[i][j][r] < votes[i][j][r+1]:
					thinVotes[i][j][r] += thinVotes[i][j][r-1]"""

	thinVotes = np.sum( thinVotes, 2 )
	scaleFactor = int( 255 / np.max( thinVotes ) )
	thinVotesPrint = np.zeros( (thinVotes.shape[0], thinVotes.shape[1], 3) )

	for i in range( votes.shape[0] ):
		for j in range( votes.shape[1] ):
			thinVotesPrint[i][j] = [ 255 - thinVotes[i][j] * scaleFactor ] * 3

	writeAndDisplayOutput(thinVotesPrint, 'votes')

	centerx, centery, radius = np.unravel_index( votes.argmax(), votes.shape )
	return (centerx, centery), radius
					

dirs = np.array( [(1,0), (1,1), (0,1), (-1,1)], dtype=int ) #directions

def thinEdges(img, edges, sobel):
	for i in range(sobel.shape[0]-1):
		for j in range(sobel.shape[1]-1):
			for c in range(sobel.shape[2]):
					if edges[i][j] == 1:
						theta = convertTheta( sobel[i][j][c][1] )
						compare1 = sobel[ i + dirs[theta][0], j + dirs[theta][1], c, 0 ] 
						#compare1 = sobel[*np.add( (i,j), dirs[theta] ), c, 0]
						compare2 = sobel[ i - dirs[theta][0], j - dirs[theta][1], c, 0 ]
						if sobel[i][j][c][0] < compare1 or sobel[i][j][c][0] < compare2:
							edges[i][j] = 0
						else:
							img[i][j] = [255, 0 , 0]

	return img, edges

def fatedges(img, sobel):
	edges = np.zeros( img.shape[0:2] )
	idxs = np.where( sobel[:,:,0,0] > 120 )
	edges[ idxs ] = 1
	img[ idxs ] = [255, 0, 0]
	
	return img, edges


def sobelMask(img):
	sobel = np.zeros((img.shape[0], img.shape[1], img.shape[2], 2))
	sx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	sy = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

	for i in range(1, img.shape[0]-1):
		for j in range(1, img.shape[1]-1):
			for c in range(img.shape[2]):
				gx = np.tensordot(img[i-1:i+2, j-1:j+2, c], sx)
				gy = np.tensordot(img[i-1:i+2, j-1:j+2, c], sy)
				M = np.sqrt(gx*gx + gy*gy)
				D = atan2(gx, gy)
				sobel[i][j][c] = [M, D]

	return sobel

def convertTheta(t):
	return int( (abs( t ) / 3.142 )*4 ) #scales from 0-pi to 0-3

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
	img[ img == 1 ] = [255, 0, 0]
	return img

def frange(start, stop, step):
	i = start
	terminate = stop-(step/10)
	while i < terminate:
		yield i
		i += step

def writeAndDisplayOutput(img, ext): #ext is what filename should end with
	outfileName = sys.argv[1] + "_" + ext + ".ppm"
	colors = 3		
	maxColor = 255

	outfile = open(outfileName, 'w')
	outfile.write('P3\n')
	outfile.write( str( img.shape[1] ) + " " + str( img.shape[0] ) + '\n')
	outfile.write( '%s\n'%maxColor )

	for i in range( img.shape[0] ):
		for j in range( img.shape[1] ):
			for k in range( colors ):
				try:
					outfile.write('%d '%img[i][j][k])
				except IndexError:
					outfile.write('%d '%img[i][j][0])

	processes.add(subprocess.Popen("display %s"%outfileName, shell=True))

if __name__ == "__main__":
	main()	