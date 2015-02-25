#####################
#					#
#	Rohan Pandit	#
#	2/11/15			#
#	Period 1        #
#					#
#####################

from random import random, uniform

def puzzle1():
	#break stick
	breaks = sorted([0, random(), random(), 1])

	for i in range(1, len(breaks)):
		if breaks[i]-breaks[i-1] > 0.5:
			return False
	return True

def puzzle2():
	#break stick
	breaks = [0, random(), 1]

	if breaks[1] > .5:
		breaks.insert(1, uniform(0, breaks[1]))
	else:
		breaks.insert(2, uniform(breaks[1], 1))

	#Determine if any piece is bigger than 0.5
	for i in range(1, len(breaks)):
		if breaks[i]-breaks[i-1] > 0.5:
			return False
	return True

def puzzle3():
	#break stick
	breaks = [0, random(), 1]

	if random() > .5:
		breaks.insert(1, uniform(0, breaks[1]))
	else:
		breaks.insert(2, uniform(breaks[1], 1))

	#Determine if any piece is bigger than 0.5
	for i in range(1, len(breaks)):
		if breaks[i]-breaks[i-1] > 0.5:
			return False
	return True

def puzzle4():
	#break stick
	breaks = [0, random(), 1]

	if random() < breaks[1]:
		breaks.insert(1, uniform(0, breaks[1]))
	else:
		breaks.insert(2, uniform(breaks[1], 1))

	#Determine if any piece is bigger than 0.5
	for i in range(1, len(breaks)):
		if breaks[i]-breaks[i-1] > 0.5:
			return False

	return True

def main():
	triangleFormed = 0
	total = 0

	for i in range(10000000): #num trials
		if puzzle4():
			triangleFormed += 1
		total += 1

	print("Puzzle 4: Probability of forming triangle: %s"%(round(triangleFormed/total,3)))

if __name__ == "__main__":
	main()

# Puzzle 1 answer: 0.250
# Puzzle 2 answer: 0.387
# Puzzle 3 answer: 0.193
# Puzzle 4 answer: 0.250

	