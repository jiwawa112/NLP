#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')
# print(df.head())

int_put = df[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
y = df['Species']
# print(int_put)
# print(y)

# y = y.replace(['setosa','versicolor'],[0,1])
# print(y)

le = LabelEncoder()
y_label = le.fit_transform(y)
# print(y_label)

scaler = StandardScaler()
int_put = scaler.fit_transform(int_put)

X_train,X_test,y_train,y_test = train_test_split(int_put,y_label,test_size=0.2,random_state=42)

clf = LogisticRegression()
clf.fit(X_train,y_train)
print(clf.intercept_,clf.coef_)

pred = clf.predict(X_test)
print('Accuracy from sk-learin: {0}'.format(clf.score(X_test,y_test)))