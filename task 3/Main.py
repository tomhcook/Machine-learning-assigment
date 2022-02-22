import numpy
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
def graph(data):
    patient = data.loc[data['Participant Condition'] == "Patient"]
    patient = patient[["Alpha"]]

    control = data.loc[data['Participant Condition'] != "Patient"]

    control = control[["Alpha"]]

    colum = [patient, control]

    plt.boxplot(colum)

    plt.show()
    patient = data_file.loc[data['Participant Condition'] == "Patient"]
    patient = patient[["Beta"]]

    control = data_file.loc[data['Participant Condition'] != "Patient"]
    control = control[["Beta"]]
    fig = sns.kdeplot(patient["Beta"], shade=True, color="r")
    fig = sns.kdeplot(control["Beta"], shade=True, color="g")

    plt.legend((' patient', ' control ', '$x$'), loc='lower right')
    plt.show()
def sksplit(data,k):
    kf = KFold(n_splits=k,shuffle=True)
    result = [(train_index, test_index) for train_index, test_index in kf.split(data)]
    return result
data_file = pd.read_csv('Task3 - dataset - HIV RVG.csv')

graph(data_file)
Tc = ['Participant Condition']
Prd = list(set(list(data_file.columns))-set(Tc))
data_file[Prd] = data_file[Prd]/data_file[Prd].max()
print(data_file.describe().transpose())
X = data_file[Prd].values
y = data_file[Tc].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=40) # data splitter

print(X_train.shape)
print(X_test.shape)
print(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='lbfgs', max_iter=500)


mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
print(confusion_matrix(y_train,predict_train))


print(classification_report(y_train,predict_train))

clf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=10)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy 10 leaf :",metrics.accuracy_score(y_test, y_pred))
clf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=5)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy 5 leaf :",metrics.accuracy_score(y_test, y_pred))
## 3.3 below
k = 10
df = data_file.drop(labels=['Image number','Bifurcation number','Artery (1)/ Vein (2)'], axis=1)
out = df["Participant Condition"].transpose()
print(df)
inp = df.drop(['Participant Condition'], axis=1)
kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)
inp = inp.transpose()
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index],inp[test_index]
    y_train, y_test = out[train_index], out[test_index]

X_train = X_train.transpose()
X_test = X_test.transpose()
y_train = y_train.transpose()
y_test = y_test.transpose()

AnnClassOne = MLPClassifier(hidden_layer_sizes=(50,50), activation='logistic', solver='adam', max_iter=5000)
AnnClassTwo = MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='adam', max_iter=5000)
AnnClassThree = MLPClassifier(hidden_layer_sizes=(1000,1000), activation='logistic', solver='adam', max_iter=5000)
TreeForestOne = RandomForestClassifier(n_estimators=50,min_samples_leaf=10)
TreeForestTwo = RandomForestClassifier(n_estimators=500,min_samples_leaf=10)
TreeForestThree = RandomForestClassifier(n_estimators=10000,min_samples_leaf=10)

AS = []
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    AnnClassOne.fit(X_train,y_train)
    predict_train = AnnClassOne.predict(X_train)
    predict_test = AnnClassOne.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))
scores = cross_val_score(AnnClassOne, inp.transpose(), out.transpose(), cv=5)
print('Average accuracy 50 ANN')
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))
AS = []
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    AnnClassTwo.fit(X_train,y_train)
    predict_train = AnnClassTwo.predict(X_train)
    predict_test = AnnClassTwo.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))
print('Average accuracy 500 ANN')
scores = cross_val_score(AnnClassTwo, inp.transpose(), out.transpose(), cv=10)
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))

AS = []
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    AnnClassThree.fit(X_train,y_train)
    predict_train = AnnClassThree.predict(X_train)
    predict_test = AnnClassThree.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))

print('Average accuracy 10000 ANN')
scores = cross_val_score(AnnClassThree, inp.transpose(), out.transpose(), cv=5)
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))

for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    TreeForestOne.fit(X_train,y_train)
    predict_train = TreeForestOne.predict(X_train)
    predict_test = TreeForestOne.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))

print('Average accuracy Tree 50')
scores = cross_val_score(TreeForestOne, inp.transpose(), out.transpose(), cv=5)
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    TreeForestTwo.fit(X_train,y_train)
    predict_train = TreeForestTwo.predict(X_train)
    predict_test = TreeForestTwo.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))

print('Average accuracy Tree 500')
scores = cross_val_score(TreeForestTwo, inp.transpose(), out.transpose(), cv=5)
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))
for train_index,test_index in kf.split(df):
    X_train, X_test = inp[train_index], inp[test_index]
    y_train, y_test = out[train_index], out[test_index]
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    y_train = y_train.transpose()
    y_test = y_test.transpose()
    TreeForestThree.fit(X_train,y_train)

    predict_train = TreeForestThree.predict(X_train)
    predict_test = TreeForestThree.predict(X_test)
    AS.append(accuracy_score(y_train, predict_train))
print('Average accuracy Tree 10000')
scores = cross_val_score(TreeForestThree, inp.transpose(), out.transpose(), cv=5)
print("CV accuracy = %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("Normal Accuracy = ",sum(AS)/len(AS))