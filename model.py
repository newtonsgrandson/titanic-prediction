from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

os.chdir("/home/muhammed/PycharmProjects/titanic-prediction")
print("Completed!")
def myEnsemble():
    X = pd.read_csv("modifiedTrain.csv", index_col = 0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    test = pd.read_csv("modifiedTest.csv", index_col=0)

    indexes = pd.read_csv("test.csv").loc[:, "PassengerId"]

    # Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    predictionsLogistic = model.predict(test)
    predictionsLogistic = pd.Series(predictionsLogistic, name="Logistic")

    # Polynomial Regression
    polyReg = PolynomialFeatures(degree=7)
    xPoly = polyReg.fit_transform(X)
    testPoly = polyReg.fit_transform(test)
    linReg = LinearRegression()
    linReg.fit(xPoly, y)
    predictionsPol = linReg.predict(testPoly)
    predictionsPol =  classification(predictionsPol)
    predictionsPol = pd.Series(predictionsPol, name="Pol")

    #Support Vector Regression
    sc1 = StandardScaler()
    xScaled = sc1.fit_transform(X)
    sc2 = StandardScaler()
    yScaled = np.ravel(sc2.fit_transform(np.array(y).reshape(-1, 1)))

    svrReg = SVR(kernel = "rbf")
    svrReg.fit(xScaled, yScaled)
    predictionsSVR = svrReg.predict(test)
    predictionsSVR = classification(predictionsSVR)
    predictionsSVR = pd.Series(predictionsSVR, name="SVR")

    tablePredictions = pd.concat([predictionsLogistic, predictionsPol, predictionsSVR], axis = 1)
    print(tablePredictions)

    predictions = chooseTrueValue(predictionsLogistic, predictionsPol, predictionsSVR)
    predictions = pd.Series(predictions, name="Survived", index=indexes)
    print(predictions)
    predictions.to_csv("/home/muhammed/PycharmProjects/titanic-prediction")

@ignore_warnings(category=ConvergenceWarning)
def calculateCombineModelMAE():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X = X_train
    y = y_train
    test = X_test

    indexes = pd.read_csv("test.csv").loc[:, "PassengerId"]

    # Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    predictionsLogistic = model.predict(test)
    predictionsLogistic = pd.Series(predictionsLogistic, name="Logistic")

    # Polynomial Regression
    polyReg = PolynomialFeatures(degree=7)
    xPoly = polyReg.fit_transform(X)
    testPoly = polyReg.fit_transform(test)
    linReg = LinearRegression()
    linReg.fit(xPoly, y)
    predictionsPol = linReg.predict(testPoly)
    predictionsPol = classification(predictionsPol)
    predictionsPol = pd.Series(predictionsPol, name="Pol")

    # Support Vector Regression
    sc1 = StandardScaler()
    xScaled = sc1.fit_transform(X)
    sc2 = StandardScaler()
    yScaled = np.ravel(sc2.fit_transform(np.array(y).reshape(-1, 1)))

    svrReg = SVR(kernel="rbf")
    svrReg.fit(xScaled, yScaled)
    predictionsSVR = svrReg.predict(test)
    predictionsSVR = classification(predictionsSVR)
    predictionsSVR = pd.Series(predictionsSVR, name="SVR")

    tablePredictions = pd.concat([predictionsLogistic, predictionsPol, predictionsSVR], axis=1)
    print(tablePredictions)

    predictions = chooseTrueValue(predictionsLogistic, predictionsPol, predictionsSVR)
    print(mean_absolute_error(y_test, predictions))

def chooseTrueValue(predictions1, predictions2, predictions3):
    predictions1 = pd.Series(predictions1)
    predictions2 = pd.Series(predictions2)
    predictions3 = pd.Series(predictions3)
    totalPredictions = [predictions1.iloc[i] + predictions2.iloc[i] + predictions3.iloc[i] for i in range(predictions1.__len__())]
    totalPredictions = [1 if i >= 2 else 0 for i in totalPredictions]
    return totalPredictions

def classification(values):
    newValues = [1 if i>=0.5 else 0 for i in values]
    return newValues

def sub_lists(my_list):
    subs = []
    for i in range(0, len(my_list) + 1):
        temp = [list(x) for x in combinations(my_list, i)]
        if len(temp) > 0:
            subs.extend(temp)

    subs.remove([])
    return subs

import numpy as np

@ignore_warnings(category=ConvergenceWarning)
def main1():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    X = X.loc[:, ['pClass0', 'pClass1', 'Male', 'Age', 'SibSp', 'Cabin5', 'Cabin1']]
    print(X)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    test = pd.read_csv("modifiedTest.csv", index_col=0).loc[:, ['pClass0', 'pClass1', 'Male', 'Age', 'SibSp', 'Cabin5', 'Cabin1']]
    indexes = pd.read_csv("test.csv").loc[:, "PassengerId"]

    index = 42
    print(index, min)
    model = LogisticRegression(random_state=index)
    model.fit(X, y)
    predictions = model.predict(test)
    predictions = pd.Series(predictions, name = "Survived", index=indexes)
    predictions.to_csv("C:/Users/AZAD KENOBI/Desktop/predictions.csv")

@ignore_warnings(category=ConvergenceWarning)
def mae():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X.loc[:, ['pClass0', 'pClass1', 'Male', 'Age', 'SibSp', 'Cabin5', 'Cabin1']], y, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(mean_absolute_error(y_test, predictions))

def randomForest():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    test = pd.read_csv("modifiedTest.csv", index_col=0)
    indexes = pd.read_csv("test.csv").loc[:, "PassengerId"]

    #RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    predictions = model.predict(test)
    predictions = pd.Series(classification(predictions), name="Survived", index=indexes)
    predictions.to_csv("predictions.csv")

def allPropogation():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    col = X.columns
    combinations = sub_lists(col)
    allMAE = []
    for i in combinations:
        X_train, X_test, y_train, y_test = train_test_split(X.loc[:,i], y, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        allMAE.append(mean_absolute_error(y_test, predictions))

    print(allMAE)
    print(pd.Series(allMAE).idxmax())

def getOptCombination():
    X = pd.read_csv("modifiedTrain.csv", index_col=0)
    col = X.columns
    combinations = sub_lists(col)
    print(combinations[4291])

@ignore_warnings(category=ConvergenceWarning)
def main2():
    import statsmodels.api as sm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X = pd.read_csv("trainData.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]

    X = X.drop(X.loc[:,"Parch"])
    print(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    X.corr()
randomForest()