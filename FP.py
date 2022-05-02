import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split


X = pd.read_csv("modifiedTrain.csv", index_col=0).loc[:, ['pClass0', 'pClass1', 'Male', 'Age', 'SibSp', 'Cabin5', 'Cabin1']]
y = pd.read_csv("train.csv").loc[:, "Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 590)
model = sm.OLS(y,X).fit()
print(model.summary())