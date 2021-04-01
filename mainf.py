# BT18CSE004 ---- NITIN PATLE
# BT18CSE047 ---- AYUSH MOHARE

# The purpose of this project is to predict the house price using features
# As well as plotting and doing analysis of results

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# real csv file using pandas , header index is 0
real_estate = pd.read_csv('Real_estate.csv', header=0)
# print(real_estate.keys())
# keys in csv file are
# (['No', 'X1 transaction date', 'X2 house age',
#        'X3 distance to the nearest MRT station',
#        'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
#        'Y house price of unit area'])


# using 1 dimensional feature i.e. house age
# will use linear regression on this single dimension
real_estate_X = real_estate.values[:, np.newaxis, 2] #using 1 dimensional feature
# our target is house price per unit area
real_estate_Y = real_estate.values[:, np.newaxis, 7] #target
# print(real_estate_X)
# first use train data using starting 30 rows
real_estate_X_train = real_estate_X[-30:] #starting 30
real_estate_Y_train = real_estate_Y[-30:]
# use last 20 rows for testing/predicting  the model
real_estate_X_test = real_estate_X[:-20] #last 20
real_estate_Y_test = real_estate_Y[:-20]

# linear model using single feature

model = linear_model.LinearRegression()
# model.fit will update coefficient values (weights) and intercept
# indirectly will make the model
model.fit(real_estate_X_train, real_estate_Y_train)

# predict the house price values using last 20 testing values
real_estate_Y_predicted = model.predict(real_estate_X_test)

# mean square error
# i. e. avg of sum of squares
print("mean squared error is : ", mean_squared_error(real_estate_Y_test, real_estate_Y_predicted))

# plot weight intercepts
print("weight: ", model.coef_)
print("intercept: ", model.intercept_)

plt.scatter(real_estate_X_test, real_estate_Y_test)
plt.plot(real_estate_X_test, real_estate_Y_predicted)
plt.show()

# Now we will use more values instead of just 30 values
# we will use 200 values
# and find mean squared error and it will become less
real_estate_X_train_2 = real_estate_X[-200:] #starting 200
real_estate_Y_train_2 = real_estate_Y[-200:]

model2 = linear_model.LinearRegression()
model2.fit(real_estate_X_train_2, real_estate_Y_train_2)
real_estate_Y_predicted_2 = model2.predict(real_estate_X_test)

# mean square error
# i. e. avg of sum of squares
print("mean squared error is : ", mean_squared_error(real_estate_Y_test, real_estate_Y_predicted_2))

# plot weight intercepts
print("weight: ", model2.coef_)
print("intercept: ", model2.intercept_)

plt.scatter(real_estate_X_test, real_estate_Y_test)
plt.plot(real_estate_X_test, real_estate_Y_predicted_2)
plt.show()

# now will will use multiple features to again reduce mean squared error
# and to make model more appropriate
# can't make plots here as features are 5

real_estate_X_3 = real_estate.iloc[:, 2:7] # using 5 dimensions
# print(real_estate_X_3)
# print(real_estate_X_3)
real_estate_X_3_train = real_estate_X_3[-200:] # starting 200
real_estate_Y_train_3 = real_estate_Y[-200:]

real_estate_X_3_test = real_estate_X_3[:-20]
real_estate_Y_test_3 = real_estate_Y[:-20]
#
model3 = linear_model.LinearRegression()
model3.fit(real_estate_X_3_train, real_estate_Y_train_3)
real_estate_Y_predicted_3 = model3.predict(real_estate_X_3_test)

# mean square error
# i. e. avg of sum of squares
print("mean squared error is : ", mean_squared_error(real_estate_Y_test_3, real_estate_Y_predicted_3))

# plot weight intercepts
print("weight: ", model3.coef_)
print("intercept: ", model3.intercept_)

