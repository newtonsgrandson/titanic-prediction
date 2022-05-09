from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import pandas as pd

@ignore_warnings(category=ConvergenceWarning)
def main():
    X = pd.read_csv("modifiedTrain.csv")
    y = pd.read_csv("train.csv").loc[:, "Survived"]

    pos = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=pos)

    # KNN
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    maeKNN = mean_absolute_error(y_test, predictions)
    print("MAE for KNN:", maeKNN)

    #LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    maeLogR = mean_absolute_error(y_test, predictions)
    print("MAE for Logistic Regression:", maeLogR)

    #Confusion Matrix
    cm = confusion_matrix(predictions, y_test)

    truePos = cm[1, 1]
    falseNeg = cm[1, 0]
    falsePos = cm[0, 1]
    trueNeg = cm[0, 0]

    all = truePos + trueNeg + falsePos + falseNeg
    yes = truePos + trueNeg
    no = falsePos + falseNeg
    predictedYES = predictions[predictions == 1].__len__()
    predictedNO = predictions[predictions == 0].__len__()

    print("True Positive:", truePos)
    print("False Negative:", falseNeg)
    print("False Positive:", falsePos)
    print("True Negative:", trueNeg)

    accuracy = (truePos + trueNeg) / (all)
    missClassification = (falseNeg + falsePos) / (all)
    truePosRate = (truePos) / (yes)
    falsePosRate = (falsePos) / (no)
    precision =  (truePos) / (predictedYES)
    F1Score = (2 * truePos) / (2 * truePos + falsePos + falseNeg)

    print("Accuracy:", accuracy)
    print("Miss Classification:", missClassification)
    print("True Positive Rate:", truePosRate)
    print("False Positive Rate:", falsePosRate)
    print("Precision:", precision)
    print("F1 Score:", F1Score)

    #ZeroR
    logic = ((cm[0, 0] + cm[0, 1]) > (cm[0, 1] + cm[1, 0]))
    if logic == 1:
        newTrue = cm[0, 0] + cm[1, 0]
        newFalse = cm[1, 0] + cm[1, 1]
        accuracyZeroR = (newTrue) / (newTrue + newFalse)
    elif logic == 0:
        newTrue = cm[0, 0] + cm[0, 1]
        newFalse = cm[1, 0] + cm[1, 1]
        accuracyZeroR = (newTrue) / (newTrue + newFalse)

    print("Zero R algorithm accuracy:", accuracyZeroR)
if __name__ == "__main__":
    main()