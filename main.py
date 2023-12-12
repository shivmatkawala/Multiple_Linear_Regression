import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
plt.style.use("seaborn-v0_8")


# Step-1 Generate Data

X, y = make_regression(n_samples=500, n_features=10, n_informative=5, noise=25.8, random_state=0)
print(X.shape, y.shape)
n_features = X.shape[1]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
print(pd.DataFrame(X).head())


def normalize(X):
    u = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - u) / std


X = normalize(X)
print(pd.DataFrame(X).head())
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

print(X.mean(axis=0))
print(X.std(axis=0))

# Visualize y as a function of each feature
for f in range(0, 10):
    plt.subplot(4, 3, f + 1)
    plt.scatter(X[:, f], y)
plt.show()

# Split the Training and Testing Data..
XT, Xt, yT, yt = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)
print(XT.shape, yT.shape)
print(Xt.shape, yt.shape)


# Modeling / Linear Regression with Multiple Features

def preprocess(X):
    if X.shape[1] == n_features:
        m = X.shape[0]
        ones = np.ones((m, 1))
        X = np.hstack((ones, X))
    return X


def hypothesis(X, theta):
    return np.dot(X, theta)


def loss(X, y, theta):
    yp = hypothesis(X, theta)
    error = np.mean((yp - y) ** 2)
    return error


def gradient(X, y, theta):
    yp = hypothesis(X, theta)
    grad = np.dot(X.T, (yp - y))
    m = X.shape[0]
    return grad / m


def Train(X, y, learning_rate=0.1, max_iters=100):
    n = X.shape[1]
    theta = np.random.randn(n)
    error_list = []
    for i in range(max_iters):
        e = loss(X, y, theta)
        error_list.append(e)
        grad = gradient(X, y, theta)
        theta = theta - learning_rate * grad

    plt.plot(error_list)
    plt.show()
    return theta


theta = Train(XT, yT)
print(theta)
print(plt.style.available)


def R2_score(y, yp):
    y_mean = y.mean()
    numerator = np.sum((y - yp) ** 2)
    denominator = np.sum((y - y_mean) ** 2)
    return 1 - (numerator / denominator)


yp = hypothesis(Xt, theta)
print(f"R2_score from scratch code:-{R2_score(yt, yp)}")


# This same cmodel can be trained by using sklearn module...
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(XT, yT)
model.predict(Xt)

print(f"R2_score from sklearn module:-{model.score(Xt, yt)}")
print(model.intercept_)
print(model.coef_)
print(model.intercept_.shape)
print(model.coef_.shape)
