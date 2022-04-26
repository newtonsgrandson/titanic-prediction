import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
        return 4
    elif "a" in allocationCabin:
        return 7
    else:
        return 4

def distNanValuesFCategorical(data, unique):
    sizeUnique = unique.__len__()
    count = 0
    for i in range(data.__len__()):
        if str(data.iloc[i]) == "nan" or str(data.iloc[i]) == "NaN":
            uniqueIndex = count % sizeUnique
            data.iloc[i] = unique.iloc[uniqueIndex]
    return data

def setDummyValuesToNumeric(data):
    mean = data.mean()
    data = data.fillna(mean)
    return data

#data = pd.read_csv("train.csv")
data = pd.read_csv("train.csv")
###########PREPROCESSING#########
#PCLASS PREPEARING
encoder1 = LabelEncoder()
#Pclass
pClass = oneHotEncodeColumn(data.loc[:, "Pclass"], ["pClass0", "pClass1", "pClass2"])
#gender
gender = oneHotEncodeColumn(data.loc[:, "Sex"], ["Male", "Female"])
#Cabin
cabin = data.loc[:,"Cabin"]
newCabinCol = [allocationCabin(cabin.iloc[i]) for i in range(cabin.__len__())]
cabin = pd.Series(newCabinCol, name="Cabin")
#Age
age = setDummyValuesToNumeric(data.loc [:, "Age"])
#Embarked
embarked = data.loc[:, "Embarked"]
embarked = distNanValuesFCategorical(embarked, pd.Series(embarked.unique()).iloc[:embarked.unique().__len__() - 2])
embarked = oneHotEncodeColumn(embarked, ["SEmberked", "CEmberked", "QEmberked"])
#SibSp
sibp = setDummyValuesToNumeric(data.loc[:,"SibSp"])
#Parch
parch = setDummyValuesToNumeric(data.loc[:, "Parch"])
#Fare
fare = setDummyValuesToNumeric(data.loc[:,"Fare"])
dataNew = pd.concat([pClass, gender, cabin, embarked, age, sibp, parch, fare], axis=1)
dataNew.to_csv("dataNew1.csv")