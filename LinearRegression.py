import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

#a function to create a linear dataset
def create_dataset(hm =20, val =1, variance = 10, step =1, correlation=False ):
	y=[]
	for i in range(hm):
		temp = val + random.randrange(-variance, variance)
		y.append(temp)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	x = [i for i in range(len(y))]
	return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)

x, y = create_dataset(20,1,20,2,'pos')
print(x,y)