from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_regression
import numpy as np 
from sklearn import tree
import sklearn.metrics as metrics
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('./src/')

from HHCART import HouseHolderCART, MSE, MeanSegmentor
from CO2 import ContinuouslyOptimizedObliqueRegressionTree
from BUTIF import BUTIF 
from RandCART import Rand_CART
from RidgeCART import Ridge_CART
from OC1Regression import OC1_Regression
from NonLinearDTRegr import NonLinear_DTRegr
from lineartree import LinearTreeRegressor

## Load data
# X_Data, y_Data = fetch_california_housing(return_X_y=True)
X_Data, y_Data = make_regression(n_samples=1000, n_features=10, noise=1, random_state=42)
X = X_Data[0:750, :]
Y = y_Data[0:750]
X_test = X_Data[750:1000, :]
Y_test = y_Data[750:1000]

## set tree depth
treeDepth = 2

### Various Variants of Decision Tree 
## CART Regression Tree from Scikit-Learn
startTime = time()
CART_model = tree.DecisionTreeRegressor(max_depth=treeDepth, min_samples_leaf=1)
CART_model = CART_model.fit(X, Y)
y_pred_train = CART_model.predict(X)
y_pred_test = CART_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("Scikit-CART: Train_r2 {}; Test_r2 {} ".format(r2_score_train, r2_score_test))
print("Scikit-CART elapsed time: {}\n".format(elapsedTime))

## HHCART Regression Tree
startTime = time()
HHCART_model = HouseHolderCART(MSE(), MeanSegmentor(), max_depth=treeDepth)
HHCART_model.fit(X, Y)
y_pred_train = HHCART_model.predict(X)
y_pred_test = HHCART_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("HH-CART r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("HH-CART elapsed time: {}\n".format(elapsedTime))

## CO2 Regression Tree
startTime = time()
CO2_model = ContinuouslyOptimizedObliqueRegressionTree(MSE(), MeanSegmentor(), thau=500, max_iter=100, max_depth=treeDepth)
CO2_model.fit(X, Y)
y_pred_train = CO2_model.predict(X)
y_pred_test = CO2_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("CO2 r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("CO2 elapsed time: {}\n".format(elapsedTime))

## BUTIF Regression Tree
startTime = time()
leaf_maxNum = 2**treeDepth
BUTIF_model = BUTIF(linear_model=LogisticRegression(max_iter=10000), task="regression", max_leaf=leaf_maxNum)
BUTIF_model.fit(X, Y)
y_pred_train = BUTIF_model.predict(X)
y_pred_test = BUTIF_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("BUTIF r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("BUTIF elapsed time: {}\n".format(elapsedTime))

## RandCART Regression Tree
startTime = time()
RandCART_model = Rand_CART(MSE(), MeanSegmentor(), max_depth=treeDepth)
RandCART_model.fit(X, Y)
y_pred_train = RandCART_model.predict(X)
y_pred_test = RandCART_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("RandCART r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("RandCART elapsed time: {} \n".format(elapsedTime))


## RidgeCART Regression Tree
startTime = time()
RidgeCART_model = Ridge_CART(MSE(), MeanSegmentor(), max_depth=treeDepth)
RidgeCART_model.fit(X, Y)
y_pred_train = RidgeCART_model.predict(X)
y_pred_test = RidgeCART_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("RidgeCART r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("RidgeCART elapsed time: {}\n".format(elapsedTime))

## OC1 Regression Tree
startTime = time()
OC1Reg_model = OC1_Regression(criterion="mse", max_depth=treeDepth)
OC1Reg_model.fit(X, Y)
y_pred_train = OC1Reg_model.predict(X)
y_pred_test = OC1Reg_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("OC1Reg r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("OC1Reg elapsed time: {}\n".format(elapsedTime))
        
## NonLinear Regression Tree
startTime = time()
NonLinearDTRegr_model = NonLinear_DTRegr(criterion="mse", max_depth=treeDepth)
NonLinearDTRegr_model.fit(X, Y)
y_pred_train = NonLinearDTRegr_model.predict(X)
y_pred_test = NonLinearDTRegr_model.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("NonLinearDTRegr r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("NonLinearDTRegr elapsed time: {}\n".format(elapsedTime))

## LinearTree 
startTime = time()
LinearRegr = LinearTreeRegressor(base_estimator=LinearRegression(), max_depth=treeDepth)
LinearRegr.fit(X, Y)
y_pred_train = LinearRegr.predict(X)
y_pred_test = LinearRegr.predict(X_test)
r2_score_train = metrics.r2_score(Y, y_pred_train)
r2_score_test = metrics.r2_score(Y_test, y_pred_test)
elapsedTime = time()-startTime
print("LinearRegr r2_Train: {}; r2_Test: {}".format(r2_score_train, r2_score_test))
print("LinearRegr elapsed time: {}\n".format(elapsedTime))
