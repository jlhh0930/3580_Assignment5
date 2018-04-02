from __future__ import print_function

import os
import subprocess
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn import datasets 

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_train.csv')
df_train = pd.read_csv(filename)

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_test.csv')
df_test = pd.read_csv(filename)

def transform_sex_to_int(value):
    if value == 'male':
        return 0
    elif value == 'female':
        return 1
def transform_emb_to_int(value):
    if value == 'C':
        return 0
    elif value == 'S':
        return 1
    elif value == 'Q':
        return 2
df_train['sex'] = df_train[['Sex']].applymap(transform_sex_to_int)
df_test['sex']=df_test[['Sex']].applymap(transform_sex_to_int)
df_train['emb'] = df_train[['Embarked']].applymap(transform_emb_to_int)
df_test['emb'] = df_test[['Embarked']].applymap(transform_emb_to_int)
df_train['Pclass'].fillna('2')
df_test['Pclass'].fillna('2')
df_train['emb'].fillna('1')
df_test['emb'].fillna('1')

features = ["sex", "Pclass"]
targets = df_test["Survived"].unique()

map_to_int = {name: n for n, name in enumerate(targets)}

df_test["Target"] = df_test["Survived"].replace(map_to_int)
df_train["Target"] = df_train["Survived"].replace(map_to_int)

y1 = df_test["Target"]
X1 = df_test[features]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X1, y1)

xTest = pd.DataFrame(X1)
yTest = pd.DataFrame(y1)
predictions = dt.predict(xTest)
countDT = 0
for i in range(0, len(y1.index)):
    if (predictions[i] == y1[i]):
        countDT = countDT+1

Xtest = df_test[features]
Xtrain = df_train[features]
Ytest = df_test["Target"]
Ytrain = df_train["Target"]
X_train = pd.DataFrame(Xtrain)
X_test = pd.DataFrame(Xtest)
y_train = pd.DataFrame(Ytrain)
y_test = pd.DataFrame(Ytest)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 8), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
nnPredictions = mlp.predict(X_test)
countNN = 0
for i in range(0, len(Ytest.index)):
    if (nnPredictions[i] == Ytest[i]):
        countNN = countNN+1

print("Survived Predictions")
print("Decision Tree: {}/91".format(countDT))
print("Neural Network: {}/91".format(countNN))
print()

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_train.csv')
df = pd.read_csv(filename)
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_test.csv')
df_test = pd.read_csv(filename)
df_train['sex'] = df_train[['Sex']].applymap(transform_sex_to_int)
df_test['sex']=df_test[['Sex']].applymap(transform_sex_to_int)
df_train['emb'] = df_train[['Embarked']].applymap(transform_emb_to_int)
df_test['emb'] = df_test[['Embarked']].applymap(transform_emb_to_int)
df_train['Pclass'].fillna('2')
df_test['Pclass'].fillna('2')
df_train['emb'].fillna('1')
df_test['emb'].fillna('1')

features = ['Pclass','Survived']
targets = df_test["Fare"].unique()

map_to_int = {name: n for n, name in enumerate(targets)}

df_test["Target"] = df_test["Fare"].replace(map_to_int)

y = df_test["Target"]
X = df_test[features]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

xTest = pd.DataFrame(X)
yTest = pd.DataFrame(y)
predictions = dt.predict(xTest)
countDT = 0
for i in range(0, len(y.index)):
    if ((predictions[i] >= y[i]-10.0) & (predictions[i] <= y[i]+10.0)):
        countDT = countDT+1

Xtest = df_test[features]
Xtrain = df_train[features]
Ytest = df_test["Target"]
Ytrain = df_train["Target"]
X_train = pd.DataFrame(Xtrain)
X_test = pd.DataFrame(Xtest)
y_train = pd.DataFrame(Ytrain)
y_test = pd.DataFrame(Ytest)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 8), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
nnPredictions = mlp.predict(X_test)
countNN = 0
for i in range(0, len(Ytest.index)):
    if ((predictions[i] >= Ytest[i]-10.0) & (predictions[i] <= Ytest[i]+10.0)):
        countNN = countNN+1

dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_train.csv')
df = pd.read_csv(filename)
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'A5_test.csv')
df_test = pd.read_csv(filename)
df_train['sex'] = df_train[['Sex']].applymap(transform_sex_to_int)
df_test['sex']=df_test[['Sex']].applymap(transform_sex_to_int)
df_train['emb'] = df_train[['Embarked']].applymap(transform_emb_to_int)
df_test['emb'] = df_test[['Embarked']].applymap(transform_emb_to_int)
df_train['Pclass'].fillna('2')
df_test['Pclass'].fillna('2')
df_train['emb'].fillna('1')
df_test['emb'].fillna('1')

features = ['Pclass','Survived']
targets = df_test["Fare"].unique()

map_to_int = {name: n for n, name in enumerate(targets)}

df_test["Target"] = df_test["Fare"].replace(map_to_int)

y = df_test["Target"]
X = df_test[features]

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

xTest = pd.DataFrame(X)
yTest = pd.DataFrame(y)
predictions = dt.predict(xTest)
countDT = 0
for i in range(0, len(y.index)):
    if ((predictions[i] >= y[i]-10.0) & (predictions[i] <= y[i]+10.0)):
        countDT = countDT+1

Xtest = df_test[features]
Xtrain = df_train[features]
Ytest = df_test["Target"]
Ytrain = df_train["Target"]
X_train = pd.DataFrame(Xtrain)
X_test = pd.DataFrame(Xtest)
y_train = pd.DataFrame(Ytrain)
y_test = pd.DataFrame(Ytest)
countLR = 0
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
preds = list(predictions)
y_test_list = list(y_test)

for i in range(0,len(predictions)):
    if ((predictions[i] >= Ytest[i]-10.0) & (predictions[i] <= Ytest[i]+10.0)):
        countLR = countLR+1

print("Fare Predictions")
print("Decision Tree: {}/91".format(countDT))
print("Neural Network: {}/91".format(countNN))
print("Linear Regression: {}/91".format(countLR))