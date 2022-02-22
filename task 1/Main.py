import numpy
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import math
def mean_squared_error(y_tester,Regress):
    mse = np.square(np.subtract(y_tester,Regress)).mean()
    return mse
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
    #plt.plot(x_train, y_train, 'g')

    plt.plot(x_train, y_train, 'bo')
    w1=0
    if degree == 0:
        w1 = numpy.average(y_train)
    else:
         w1 = W_Pol_Fit(x_train, y_train, degree)
    Xtest1 = Pol_Matrix(x_train, degree)
    ytest1 = Xtest1.dot(w1)
    print(w1)
    print
    plt.plot(x_train, ytest1, 'r')

    plt.legend(('training points', 'poly line', '$x$'), loc = 'lower right')
    title = "polynomial regression " +  str(degree)
    plt.title(title)
    plt.savefig('polynomial.png')
    plt.show()
    return ytest1;

data_file = pd.read_csv('Task1 - dataset - pol_regression.csv')
data_file = data_file.sort_values(by=['x'], ascending=True)

x = data_file['x']
y = data_file['y']
height = len(x)
x_trainer = x[0:math.floor(height*0.7)]
y_trainer = y[0:math.floor(height*0.7)]

x_tester = x[0:math.ceil(height*0.3)]
y_tester = y[0:math.ceil(height*0.3)]

Regress0 = pol_regression(x,y,0)
Regress1 = pol_regression(x,y,1)
Regress2 = pol_regression(x,y,2)
Regress3 = pol_regression(x,y,3)
Regress4 = pol_regression(x,y,6)
Regress5 = pol_regression(x,y,10)

RsmeTest = []
RsmeTrain = []
for i in range(11):
    Regress0 = pol_regression(x_trainer,y_trainer,i)
    RsmeTrain.append(np.sqrt(mean_squared_error(y_trainer,Regress0)))
    Regress0.resize((6),refcheck=False)
    RsmeTest.append(np.sqrt(mean_squared_error(y_tester,Regress0)))

print(RsmeTrain)
print(RsmeTest)
plt.plot(RsmeTrain, color='r')
plt.xticks([0,1,2,3,6,10])
plt.yticks(np.arange(0, 80, step=5))

plt.plot(RsmeTest,color='b')
plt.legend(('Train regression','Test regression', 'poly line', '$x$'), loc='upper right')

plt.show()
