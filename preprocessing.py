import pandas as pd
from sklearn.preprocessing import LabelEncoder

pTrainData = "train.csv"
#pTrainData = "test.csv"
toCsv = "modifiedTrain.csv"
#toCsv = "modifiedTest.csv"

def binarizeData(labelEncode, uniqueSize):
    oneHotEncoded = []
    for i in labelEncode:
        encoded = []
        for j in range(uniqueSize):
            if j == i:
                encoded.append(1)
            else:
                encoded.append(0)
        oneHotEncoded.append(pd.Series(encoded))

    return pd.DataFrame(oneHotEncoded)

def oneHotEncodeColumn(column, columnNames):
    encoder1 = LabelEncoder()
    columnX = column.unique()
    encoder1.fit(columnX)
    encodedColumn = encoder1.transform(column)
    uniqueSize = columnX.__len__()
    column = binarizeData(encodedColumn, uniqueSize)
    column.columns = columnNames
    return column

def allocationCabin(allocationCabin):
    allocationCabin = str(allocationCabin).lower()
    if "g" in allocationCabin:
        return 1
    elif "f" in allocationCabin:
        return 2
    elif "e" in allocationCabin:
        return 3
    elif "d" in allocationCabin:
        return 4
    elif "c" in allocationCabin:
        return 5
    elif "b" in allocationCabin:
        return 6
    elif allocationCabin == "nan":
        return "nan"
    elif "a" in allocationCabin:
        return 7
    else:
        return "nan"

def distNanValuesFCategorical(data, unique):
    sizeUnique = unique.__len__()
    count = 0
    for i in range(data.__len__()):
        element = str(data.iloc[i])
        if element == "nan" or element == "NaN":
            uniqueIndex = count % sizeUnique
            data.iloc[i] = unique.iloc[uniqueIndex]
            count += 1
    return data

def setDummyValuesToNumeric(data):
    mean = data.mean()
    data = data.fillna(mean)
    return data

#data = pd.read_csv("train.csv")
data = pd.read_csv(pTrainData)
###########PREPROCESSING#########
#PCLASS PREPEARING
encoder1 = LabelEncoder()
#Pclass
pClass = oneHotEncodeColumn(data.loc[:, "Pclass"], ["pClass0", "pClass1", "pClass2"])
pClass.drop("pClass2", axis = 1, inplace = True)
#gender
gender = oneHotEncodeColumn(data.loc[:, "Sex"], ["Male", "Female"])
#Cabin
cabin = data.loc[:,"Cabin"]
newCabinCol = pd.Series([allocationCabin(cabin.iloc[i]) for i in range(cabin.__len__())])
newCabinCol = distNanValuesFCategorical(newCabinCol, pd.Series(newCabinCol.unique()).drop(0))
cabin = pd.Series(newCabinCol, name="Cabin")
cabin = oneHotEncodeColumn(cabin, [f"Cabin{i}" for i in cabin.unique()])
cabin.drop("Cabin7", axis = 1, inplace = True)
cabin.drop("Cabin6", axis = 1, inplace = True)
cabin.drop("Cabin2", axis = 1, inplace = True)

#Age
age = setDummyValuesToNumeric(data.loc [:, "Age"])
#Embarked
embarked = data.loc[:, "Embarked"]
embarked = distNanValuesFCategorical(embarked, pd.Series(embarked.unique()).iloc[:embarked.unique().__len__() - 2])
embarked = oneHotEncodeColumn(embarked, ["SEmberked", "CEmberked", "QEmberked"])
#SibSp
sibp = setDummyValuesToNumeric(data.loc[:,"SibSp"]) + 1
#Parch
parch = setDummyValuesToNumeric(data.loc[:, "Parch"]) + 1
#Fare
fare = setDummyValuesToNumeric(data.loc[:,"Fare"]) + 1
dataNew = pd.concat([pClass, gender, embarked, age, sibp, cabin], axis=1)
dataNew.to_csv(toCsv)