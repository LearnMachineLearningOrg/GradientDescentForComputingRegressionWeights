# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:24:55 2019

@author: rajui

1. Hypothesis function for linear regression with one feature is: y = mx + b
    Here 
    'y' is the dependent variable
    'x' is the independent variable
    'm' is the coefficient or weight and 
    'b' is the intercept

2. Most general cost function that is used in linear regression is 
    'Mean Square Error'

3. Gradient Descent is a first order iterative function that is used to find 
    the values of coefficients (for example, in the above hypothesis function 
    'm' and 'b') that will minimize the cost function
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def getOptimalWeightsUsingGradientDescent (x, y, mCurrent, bCurrent, 
                                           numberOfIterations, learningRate):
    n = len(x)
    
    for i in range(numberOfIterations):
        #The hypothesis function y = mx + b
        yPredicted = mCurrent * x + bCurrent
                                
        if i >= 1:
            previousCost = cost

        #The cost funtion, Mean Square Error function
        cost = (1/n) * sum([val**2 for val in (y-yPredicted)])        
        
        #Calculate the patial derivative w.r.t 'm' to the cost function
        mDerivativeOfCostFunction = -(2/n) * sum(x*(y-yPredicted))
        #Calculate the patial derivative w.r.t 'm' to the cost function        
        bDerivativeOfCostFunction = -(2/n) * sum(y-yPredicted)
        #Computing the current 'm' value using the learning rate and 
        #the partial derivative
        mCurrent = mCurrent - learningRate * mDerivativeOfCostFunction
        #Computing the current 'b' value using the learning rate and 
        #the partial derivative
        bCurrent = bCurrent - learningRate * bDerivativeOfCostFunction
        print ("i {}, m {}, b {}, mDerivativeOfCostFunction {}, bDerivativeOfCostFunction {}, cost {}".format(i, mCurrent, bCurrent, mDerivativeOfCostFunction, bDerivativeOfCostFunction, cost))
        
        plt.plot(x, yPredicted, color='green', alpha=0.1)

        if i >= 1:
            #Checking if the cost computed in the previous step and the 
            #current step are close enough so that we can descide whether to 
            #break from the iterations
            if math.isclose(previousCost, cost, rel_tol=1e-09, abs_tol=0.0):
                plt.plot(x, yPredicted, color='red')
                return;
                
"""
- RM       average number of rooms per dwelling
- MEDV     Median value of owner-occupied homes in $1000's
"""
#Load the boston dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
print("*** Preview the dataSet and look at the statistics of the dataSet ***")
print (boston_dataset.DESCR)

boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

#This is one of the feature from the boston dataset
numberOfRoomsArray = np.array(boston['RM'])
#This is the label in the boston dataset
priceOfHouseArray = np.array(boston['MEDV'])
plt.scatter(numberOfRoomsArray, priceOfHouseArray, marker='o')
plt.xlabel('RM')
plt.ylabel('MEDV')

#Initial values of m and b in the hypothesis function y = mx + b
mCurrent = bCurrent = 0

#Number of iterations
numberOfIterations = 15000

#Learning rate
learningRate = 0.02

#Invoke the getOptimalWeightsUsingGradientDescent function to perform the
#Gradient descent algorithm
getOptimalWeightsUsingGradientDescent (numberOfRoomsArray, priceOfHouseArray, 
                                       mCurrent, bCurrent, 
                                       numberOfIterations, learningRate)

print ("************************ Linear Regression *********************")
numberOfRooms = pd.DataFrame(np.c_[boston['RM']], columns = ['RM'])
priceOfHouse = boston['MEDV']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(numberOfRooms, priceOfHouse)

#Predicting the prices
predictedPriceOfHouse = regressor.predict(numberOfRooms)

#The coefficients / the linear regression weights
print ('Coefficients: ', regressor.coef_)

#Calculating the Mean of the squared error
from sklearn.metrics import mean_squared_error
print ("Mean squared error: ", mean_squared_error(priceOfHouse, 
                                                  predictedPriceOfHouse))

#Finding out the accuracy of the model
from sklearn.metrics import r2_score
accuracyMeassure = r2_score(priceOfHouse, predictedPriceOfHouse)
print ("Accuracy of model is {} %".format(accuracyMeassure*100))
