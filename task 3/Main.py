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
    fig = sns.kdeplot(control["Beta"], shade=True, color="g")
    fig = sns.kdeplot(patient["Beta"], shade=True, color="r")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=40)

print(X_train.shape); print(X_test.shape)
mlp = MLPClassifier(hidden_layer_sizes=(50,50), activation='relu', solver='adam', max_iter=500)
mlp2 = MLPClassifier(hidden_layer_sizes=(500,500), activation='relu', solver='adam', max_iter=500)
mlp3 = MLPClassifier(hidden_layer_sizes=(1000,1000), activation='relu', solver='adam', max_iter=500)

#mlp.fit(X_train,y_train)

#mlp2.fit(X_train,y_train)

#mlp3.fit(X_train,y_train)

#predict_train = mlp.predict(X_train)
#predict_train2 = mlp2.predict(X_train)
#predict_train3 = mlp3.predict(X_train)
#predict_test = mlp.predict(X_test)
#predict_test2 = mlp2.predict(X_test)
#predict_test3 = mlp3.predict(X_test)
#print(confusion_matrix(y_train,predict_train))


#print(classification_report(y_train,predict_train))
#print(confusion_matrix(y_train,predict_train2))
#print(classification_report(y_train,predict_train2))
#print(confusion_matrix(y_train,predict_train3))
#print(classification_report(y_train,predict_train3))
#clf = RandomForestClassifier(n_estimators=50,min_samples_leaf=10)
#clf2 = RandomForestClassifier(n_estimators=500,min_samples_leaf=10)
#clf3 = RandomForestClassifier(n_estimators=10000,min_samples_leaf=10)
#clf.fit(X_train,y_train)
#clf2.fit(X_train,y_train)
#clf3.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#y_pred2 = clf2.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
#y_pred3 = clf3.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))
k = 10
SplitData= sksplit(data_file,k)
print(SplitData[0])

#predict_train = mlp.predict(X_train)
