#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import jieba

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('chinese_news.csv')
# print(data.head())
print(data.shape)

data = data.dropna()
# print(type(data))
print(data.shape)

content = data['content']
source = data['source']
# print(content)
# print(source)

y_label = []
for i in source:
    if i == '新华社':
        i = 1
    else:
        i = 0
    y_label.append(i)
# print(y_label)
# print(len(y_label))

def token(string):
    return ' '.join(re.findall('[\w|\d]+',string))

contents = [token(str(a)) for a in content]


# count_words = CountVectorizer()
# X_train_counts = count_words.fit_transform(contents)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(contents)
# print(vectorizer.get_feature_names())
# print(X_train_tfidf.shape)
# print(X_train_tfidf)


X_train,X_test,y_train,y_test = train_test_split(X_train_tfidf,y_label,test_size=0.2,random_state=42)

clf = LogisticRegression()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)
print(clf.score(X_test,y_test))
