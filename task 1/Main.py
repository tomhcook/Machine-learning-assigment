import numpy
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

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
    print
    plt.plot(x_train, ytest1, 'r')

    plt.legend(('training points', 'ground truth', '$x$'), loc = 'lower right')

    plt.savefig('polynomial.png')
    plt.show()
    return ytest1;

data_file = pd.read_csv('Task1 - dataset - pol_regression.csv')
data_file = data_file.sort_values(by=['x'], ascending=True)

x = data_file['x']
y = data_file['y']

x_trainer = x.sample(frac=0.7,random_state=0)
y_trainer = y.sample(frac=0.7,random_state=0)

x_tester = x.drop(x_trainer.index)
y_tester = y.drop(y_trainer.index)

Regress0 = pol_regression(x_trainer,y_trainer,0)
print(Regress0)

Regress1 = pol_regression(x_trainer,y_trainer,1)

print(Regress1.shape)

print(y_tester.shape)
Regress1.resize((6),refcheck=False)
print(Regress1)
print(np.sqrt(mean_squared_error(y_tester,Regress1)))

Regress2 = pol_regression(x,y,2)
Regress2.resize((6),refcheck=False)
print(np.sqrt(mean_squared_error(y_tester,Regress2)))
Regress3 = pol_regression(x,y,3)
Regress3.resize((6),refcheck=False)
print(np.sqrt(mean_squared_error(y_tester,Regress3)))
#print(np.sqrt(mean_squared_error(y_tester,Regress3)))
Regress4 = pol_regression(x,y,6)
Regress4.resize((6),refcheck=False)
print(np.sqrt(mean_squared_error(y_tester,Regress4)))
Regress5 = pol_regression(x,y,10)
Regress5.resize((6),refcheck=False)
print(np.sqrt(mean_squared_error(y_tester,Regress5)))


