#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')
print(df.head(),'\n')
print(df.describe(),'\n')
print(df.info(),'\n')

# plot
sns.pairplot(df,hue='Species',height=2.5)
plt.show()

print(df['Species'].value_counts())
print()

# Simple Logistic Regression
final_df = df[df['Species'] !='Iris-virginica']
print(final_df.head())

# Outliter Check
sns.pairplot(final_df,hue='Species',height=2.5)
plt.show()

# SEPAL LENGTH
final_df.hist(column='Sepal.Length',bins=20,figsize=(10,5))
final_df.loc[final_df['Sepal.Length'] < 1, ['Sepal.Length']] = final_df['Sepal.Length']*100
final_df.hist(column='Sepal.Length',bins=20, figsize=(10,5))

# SEPAL WIDTH
final_df = final_df.drop(final_df[(final_df['Species'] == 'setosa')& (final_df['Sepal.Width'] < 2.5)].index)
sns.pairplot(final_df,hue='Species',height=2.5)

# Label Encoding
final_df['Species'].replace(["setosa","versicolor"], [1,0], inplace=True)
print(final_df.head())

# Model Construction
inp_df = final_df.drop(final_df.columns[[4]],axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]],axis=1)

scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)

X_train,X_test,y_train,y_test = train_test_split(inp_df,out_df,test_size=0.2,random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()
print(y_tr_arr)
print(y_ts_arr)

print('Input Shape',(X_tr_arr.shape))
print('Output Shape',X_test.shape)

def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b

def sigmoid_activation(result):
    final_result = 1 / (1 + np.exp(-result))
    return final_result

def model_optimize(w,b,X,Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid_activation(np.dot(w,X.T) + b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))

    # Gradient calculation
    dw = (1/m)*(np.dot(X.T,(final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))

    grads = {"dw":dw, "db":db}

    return grads,cost

def model_predict(w,b,X,Y,learning_rate,no_iterations):
    costs = []
    for i in range(no_iterations):

        grads,cost = model_optimize(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        # weight update
        w = w - (learning_rate * (dw.T))
        b = b- (learning_rate * db)

        if (i % 100 == 0):
            costs.append(cost)
            print("Cost after %i iteration is %f" %(i, cost))

    # final parameters
    coeff = {"w":w,"b":b}
    gradient = {"dw":dw,"db":db}

    return coeff,gradient,costs

def predict(final_pred,m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
        else:
            y_pred[0][i] = 0
    return y_pred

# Get number of features
n_features = X_tr_arr.shape[1]
print('Number of Features',n_features)
w,b = weightInitialization(n_features)

# Gradient Descent
coeff,gradient,costs = model_predict(w,b,X_tr_arr,y_tr_arr,learning_rate=0.0001,no_iterations=1000)

# Final prediction
w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept',b)

final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T) + b)
final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T) + b)

m_tr = X_tr_arr.shape[0]
m_ts = X_ts_arr.shape[0]

y_tr_pred = predict(final_train_pred,m_ts)
print('Training Accuracy',accuracy_score(y_tr_pred.T, y_tr_arr))

y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy',accuracy_score(y_ts_pred.T, y_ts_arr))

# plot

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()

clf = LogisticRegression()
clf.fit(X_tr_arr,y_tr_arr)

print(clf.intercept_,clf.coef_)

pred = clf.predict(X_ts_arr)
print('Accuracy from sk-learin: {0}'.format(clf.score(X_ts_arr,y_tr_arr)))