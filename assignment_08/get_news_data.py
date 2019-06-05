#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

categories = ['alt.atheism', 'soc.religion.christian',
             'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,shuffle=True,random_state=42)
# print(twenty_train)
text_data = twenty_train.target_names
print(text_data)

print(len(twenty_train.data))
print(len(twenty_train.filenames))

# print the first lines of the first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

print(twenty_train.target[:10])

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

# Tokenizing text
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# Training a classifier
clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_trans.transform(X_new_counts)

pred = clf.predict(X_new_tfidf)

for doc,category in zip(docs_new,pred):
    print('%r => %s ' % (doc,twenty_train.target_names[category]))


# Building a pipeline¶
text_clf = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf',MultinomialNB())
])

text_clf.fit(twenty_train.data,twenty_train.target)


# Evaluation of the performance on the test set
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories,shuffle=True,random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))



