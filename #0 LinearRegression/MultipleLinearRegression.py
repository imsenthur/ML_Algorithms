import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

style.use('dark_background')

#loading dataset
boston_data = datasets.load_boston(return_X_y = False)
X = boston_data.data
y = boston_data.target

#splitting dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.5, random_state =1)

#defining our classifier
clf = linear_model.LinearRegression()

#training our classifier
clf.fit(X_train, y_train)

#Outputing the coefficients
print("Coefficients: \n ", clf.coef_)

#accuracy of the classifier
print("Accuracy : {}".format(clf.score(X_test, y_test)))

#plotting residual errors in training data
plt.scatter(clf.predict(X_train), clf.predict(X_train) - y_train,
            color = 'y', s = 10, label = 'Train_data')
 
#plotting residual errors in test data
plt.scatter(clf.predict(X_test), clf.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test_data')
plt.legend(loc = 'upper right')
plt.title('LinearRegression')
plt.show()
