import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


def calc_mse(X, Y ,theta):
    return sum(np.square(-Y + float(theta[0][0]) + float(theta[1][0]) * X))/len(X)

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

#print(len(x_train))
y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
X = np.vstack([np.ones(len(x_train)), x_train]).T
Y = y_train[:, np.newaxis]

theta_best = np.dot((np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)),Y)
#print(theta_best)


# TODO: calculate error
MSE = calc_mse(x_test, y_test, theta_best)

print(MSE)
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0][0]) + float(theta_best[1][0]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
st_x = (x_train - np.mean(x_train))/np.std(x_train)
st_y = (y_train - np.mean(y_train))/np.std(y_train)
#st_testy = (y_test - np.mean(y_test))/np.std(y_test)
#st_testx = (x_test - np.mean(x_test))/np.std(x_test)

stand_tX = np.vstack([np.ones(len(st_x)), st_x]).T
stand_tY = st_y[:, np.newaxis]
lrn_pase = 0.0001
theta_best = [[np.random.random()],
              [np.random.random()]]
epsilon = 0.000000000000000001
# TODO: calculate theta using Batch Gradient Descent
MSE = calc_mse(st_x,st_y,theta_best)
while True:
    theta_best =theta_best - lrn_pase*2/len(st_x)*np.dot(stand_tX.T,np.dot(stand_tX, theta_best) - stand_tY)
    if np.abs(MSE - calc_mse(st_x,st_y,theta_best)) < epsilon:
        break 
    else:
        MSE = calc_mse(st_x, st_y,theta_best)
# TODO: calculate error

# plot the regression line
scaled_theta = theta_best.copy()
scaled_theta[1] = scaled_theta[1] * np.std(y_train) / np.std(x_train)
scaled_theta[0] = np.mean(y_train) - scaled_theta[1] * np.mean(x_train)
#scaled_theta = scaled_theta.reshape(-1)


print(calc_mse(x_test,y_test,scaled_theta))
x = np.linspace(min(x_test), max(x_test), 100)
y = float(scaled_theta[0][0]) + float(scaled_theta[1][0]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()