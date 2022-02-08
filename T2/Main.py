import numpy
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np


def Pol_Matrix(features_train, degree):
  X = np.ones(features_train.shape)
  for i in range(1,degree + 1):
    X = np.column_stack((X, features_train ** i))
  return X
def W_Pol_Fit(features_train,y_train,degree):
    X = Pol_Matrix(features_train, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y_train))
    return w

def pol_regression(x_train, y_train, degree):
    plt.plot(x_test, y_test, 'g')
    plt.plot(x_train, y_train, 'bo')
    w1=0
    if degree == 0:
        w1 = numpy.average(y_train)
    else:
         w1 = W_Pol_Fit(x_train, y_train, degree)
    Xtest1 = Pol_Matrix(x_test, degree)
    ytest1 = Xtest1.dot(w1)
    plt.plot(x_test, ytest1, 'r')

    plt.legend(('training points', 'ground truth', '$x$'), loc = 'lower right')

    plt.savefig('polynomial.png')
    plt.show()
    return w1;

data_file = pd.read_csv('Task1 - dataset - pol_regression.csv')
data_file = data_file.sort_values(by=['x'], ascending=True)
print(data_file)
x = data_file['x']
y = data_file['y']
print(numpy.average(y))
x_train = x.sample(frac=0.7,random_state=0)
y_train = y.sample(frac=0.7,random_state=0)

x_test = x.drop(x_train.index)
y_test = y.drop(y_train.index)

Regress0 = pol_regression(x,y,0)
print(Regress0)
Regress1 = pol_regression(x,y,1)
print(Regress1)

pol_regression(x,y,3)
pol_regression(x,y,6)
pol_regression(x,y,50)


