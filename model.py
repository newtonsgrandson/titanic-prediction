from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import numpy as np

def main1():
    X = pd.read_csv("trainData.csv", index_col=0)
    y = pd.read_csv("train.csv").loc[:, "Survived"]
    test = pd.read_csv("testData.csv", index_col=0)
    indexes = pd.read_csv("test.csv").loc[:, "PassengerId"]

    index = 590
    print(index, min)
    model = LogisticRegression(random_state=index)
    model.fit(X, y)
    predictions = model.predict(test)
    predictions = pd.Series(predictions, name = "Survived", index=indexes)
    predictions.to_csv("predictions.csv")

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

main2()