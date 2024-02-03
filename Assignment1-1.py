# Libraries you will be using in this assignment
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt

# Makes our plots look nicer
matplotlib.style.use("seaborn-v0_8-notebook")

# Let's also hide unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

# A seed for the random number generator so all results are reproducible
np.random.seed(15)

# Data Inspection
X = np.linspace(0,1,25) #linspace: (구간 시작점, 구간 끝점, 구간 내 숫자 개수)
print(X) 
Y_linear = 0.2 + X * 0.3
Y_exponential = 0.2 + 0.5 * X**2

# Exercise 1a)
# Store the shape of X into the variable X_shape. 
# Make sure that your code passes the tests! (1 Point)

X_shape = np.shape(X)
print(X_shape) #(25,)

# Exercise 1 b)
# What is the size of X? 
# Use the numpy size function and store the result in the variable X_size. (1 Point)

X_size = np.size(X)
print (X_size) #25

# Exercise 1 c)
# np.shape and np.size returned two different types. \
# Use type() to determine the types of X_size and X_shape. (1 Point)

type_X_size = type(X_size)
print(type_X_size) # <class 'int'>
type_X_shape = type(X_shape)
print(type_X_shape) # <class 'tuple'>

# Exercise 2)
# Generate an array of random normal noise and 
# add it to the linear and exponential data.

noise_variance = 0.01
noise = np.random.randn(X_size) * np.sqrt(noise_variance)
print(noise)
Y_linear_noisy = noise + Y_linear
print(Y_linear_noisy)
Y_exponential_noisy = noise + Y_exponential
print(Y_exponential_noisy)


#Exercise 3)
#Plot Y_linear_noisy and Y_exponential_noisy against their respective X values. 
# Use the plt alias for matplotlib.pyplot that we imported in the first cell of the notebook.

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axes[0].scatter(X, Y_linear_noisy, label = "Noisy", color = "red", s = 20)
axes[0].plot(X,Y_linear, label = "Noise-Free", color = "blue", lw = 2)
axes[0].set_title("Y_Linear Graph", fontsize = 14)
axes[0].set_xlabel("X", fontsize = 14)
axes[0].set_ylabel("Y_Linear", fontsize = 14)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].grid(True)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)


axes[1].scatter(X, Y_exponential_noisy, label = "Noisy", color = "red", s = 20)
axes[1].plot(X,Y_exponential, label = "Noise-Free", color = "blue", lw = 2)
axes[1].set_title("Y_exponential Graph", fontsize = 14)
axes[1].set_xlabel("X", fontsize = 14)
axes[1].set_ylabel("Y_Exponential", fontsize = 14)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Linear vs Exponential Functions with Noise', fontsize=18)

plt.show()

#Models & Model Errors
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
lin_reg_model_linear = LinearRegression()
lin_reg_model_exponential = LinearRegression()
mlp_model_linear = MLPRegressor(solver='lbfgs')
mlp_model_exponential = MLPRegressor(solver='lbfgs')

mlp_model_linear.fit(X.reshape(-1, 1), Y_linear_noisy) #MLPRegressor(solver='lbfgs')
mlp_model_exponential.fit(X.reshape(-1, 1), Y_exponential_noisy) #MLPRegressor(solver='lbfgs')
lin_reg_model_linear.fit(X.reshape(-1, 1), Y_linear_noisy) #LinearRegression()
lin_reg_model_exponential.fit(X.reshape(-1, 1), Y_exponential_noisy) #LinearRegression()
lin_reg_model_linear.coef_ #array([0.35026614])
lin_reg_model_linear.intercept_ #0.16112669615625438

#Predictions
lin_reg_pred_lin = lin_reg_model_linear.predict(X.reshape(-1,1))
mlp_pred_lin = mlp_model_linear.predict(X.reshape(-1,1))
lin_reg_pred_exp = lin_reg_model_exponential.predict(X.reshape(-1, 1))
mlp_pred_exp = mlp_model_exponential.predict(X.reshape(-1,1))

#Exercise 4)
#Now it's your go! Plot the predictions that we just calculated for all our models. 
#Plot predictions for both MLP and linear regression on both functions.
fig, axes = plt.subplots(1, 2,figsize=(12, 6), sharey=True)

# 선형 데이터 서브플롯
axes[0].scatter(X, Y_linear_noisy, label="True Data", color="gray", alpha=0.5)
axes[0].plot(X, mlp_pred_lin, label="MLP Prediction", color="red")
axes[0].plot(X, lin_reg_pred_lin, label="Linear Regression Prediction", color="blue")
axes[0].set_title("Predictions for Linear Data", fontsize=14)
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].legend()

# 지수 데이터 서브플롯
axes[1].scatter(X, Y_exponential_noisy, label="True Data", color="gray", alpha=0.5)
axes[1].plot(X, mlp_pred_exp, label="MLP Prediction", color="red")
axes[1].plot(X, lin_reg_pred_exp, label="Linear Regression Prediction", color="blue")
axes[1].set_title("Predictions for Exponential Data", fontsize=14)
axes[1].set_xlabel("X")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.suptitle('Model Predictions vs True Data', fontsize=18)

plt.show()

#Exercise 5)
#Implement the MSE in the function calculate_MSE below.
def calculate_MSE(Y_Predicted, Y_True):
    square_diff = (Y_Predicted - Y_True)**2
    MSE = np.mean(square_diff)
    return MSE

mlp_error_exponential = calculate_MSE(mlp_pred_exp, Y_exponential_noisy)
mlp_error_linear = calculate_MSE(mlp_pred_lin, Y_linear_noisy)
print(mlp_error_exponential)
print(mlp_error_linear)

lin_reg_error_exponential = calculate_MSE(lin_reg_pred_exp, Y_exponential_noisy)
lin_reg_error_linear = calculate_MSE(lin_reg_pred_lin, Y_linear_noisy)
print(lin_reg_error_exponential)
print(lin_reg_error_linear)

#Exercise 6)
#Now plot the MSE errors as a bar plot.
errors_linear = [lin_reg_error_linear, mlp_error_linear]
errors_exponential = [lin_reg_error_exponential, mlp_error_exponential]
labels = ["Linear Regression", "MLP"]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axes[0].bar(labels, errors_linear, color=['orange', 'brown'])
axes[0].set_title("MSE for Linear Data", fontsize=14)
axes[0].set_ylabel("Mean Squared Error")

axes[1].bar(labels, errors_exponential, color=['orange', 'brown'])
axes[1].set_title("MSE for Exponential Data", fontsize=14)
axes[1].set_ylabel("Mean Squared Error")

plt.suptitle('Model Error Comparison', fontsize=18)

plt.show()