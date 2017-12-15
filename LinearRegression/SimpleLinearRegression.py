import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('dark_background')

#creates a linear dataset
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

#computes the coefficients of the regression line
def compute_coeffs(x, y):
	m_x = np.mean(x)
	m_y = np.mean(y)
	n = len(x)

	SS_xy = np.sum(y*x - n*m_x*m_y)
	SS_xx = np.sum(x**2 - n*m_x**2)

	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x

	return b_0, b_1

#computes the squared error of the regression line formed
def compute_squared_error(y, reg_line):
	mean_line_y = np.mean(y)

	reg_line_sq = np.sum((y - reg_line)**2)
	mean_line_sq = np.sum((y - mean_line_y)**2)

	return 1 - (reg_line_sq/mean_line_sq)

#predicts data
def predict(x,coeffs= []):
	return x*coeffs[1] + coeffs[0]

x, y = create_dataset(20,1,20,2,'pos')
b = compute_coeffs(x, y)

regression_line = x*b[1] + b[0]
r_sq = compute_squared_error(y,regression_line)

#testing
x_test = create_dataset(10, 1, 10, 1, 'pos')
y_predict = [predict(i,b) for i in x_test]

print("The squared error of the regression line is {}".format(r_sq))
plt.scatter(x,y, color='y')
plt.plot(x,regression_line, color ='b')
plt.scatter(x_test,y_predict,color='g')
plt.show()