import numpy as np
import sys
import os

def main():
	f = open(sys.argv[1],'r').read().split() #list
	f.pop(0)
	width = f.pop(0)
	height = f.pop(0)
	maxColor = f.pop(0)

	image = np.array(f)
	image = image.resize(image, (height, width 3))
	bw = np.zeros(height,width)



def grayscale():

	for i in range(width):
		for j in range(height):
			bw[i][j] = image[i][j][0]*.1 + image[i][j][1]*.2 + image[i][j][2]*3 #grayscale

	outfile = open(sys.argv[1] + "_bw" + ".ppm", 'w')
	outfile.write('P3\n')
	outfile.write(width + " " + height + '\n')
	outfile.write(maxColor) 





if __name__ == "__main__":
	main()