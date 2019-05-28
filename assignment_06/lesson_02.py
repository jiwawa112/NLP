"""
TFIDF Key words
"""
# 1. keywords:(1)在一篇文章中出现频率很高

import re
import math
import jieba
import pandas as pd
import numpy as np
import wordcloud
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

csv_path = 'F:/python/NLP_course/NLP_course/Course_6/sqlResult_1558435.csv'
# content = pd.read_csv(csv_path,encoding='unicode_escape')
content = pd.read_csv(csv_path,encoding='gb18030')
content = content.fillna('') # 填充nan数据
news_content = content['content'].tolist()

def cut(string):return ' '.join(jieba.cut(string))
# print(cut('这是一个测试'))

def token(string):
    return re.findall(r'[\d|\w]+',string)

# print(token('这是一个测试\n\n\n'))

news_content = [token(n) for n in news_content]
news_content = [' '.join(n) for n in news_content]
news_content = [cut(n) for n in news_content]
print(news_content[0])

# 一个特定的词是否出现在一个特定的文本中(计算包含该特定词的文本数量)
def document_frequency(word):
    return sum(1 for n in news_content if word in n)
print(document_frequency('小米'))

# 逆文本频率
def idf(word):
    """gets the inversed document frequency"""
    return math.log10(len(news_content) / document_frequency(word) + 1)# 分母加1，可防止分母为0的情况

# idf('的') < idf('小米')

# 计算词频(计算的是一个特定的词在一篇特定的文本中的出现次数)
def tf(word,document):
    """
    gets the term frequency of a @word in a @document.
    """
    words = document.split()

    return sum(1 for w in words if w == word)
print(content['content'][11])
print(tf('银行', news_content[11]))
print(tf('创业板', news_content[11]))
print(idf('创业板'))
print(idf('银行'))
print(idf('短期'))
print(tf('短期', news_content[11]))

# IF-IDF
def get_keywords_of_a_document(document):
    words = document.split()

    tf_idf = [(w,tf(w,document) * idf(w)) for w in words]

    tf_idf = sorted(tf_idf,key=lambda x:x[1],reverse=True)

    return tf_idf

print(get_keywords_of_a_document(news_content[11]))
print(news_content[0])
print(news_content[11])
machine_new_keywords = get_keywords_of_a_document(news_content[101])
print(news_content[101])
print(get_keywords_of_a_document(news_content[101]))


"""
# Wordcloud
wc = wordcloud.WordCloud('F:/python/NLP_course/NLP_course/Course_6SourceHanSerifSC-Regular.otf')
machine_new_keywords_dict = {w: score for w, score in machine_new_keywords}
plt.imshow(wc.generate_from_frequencies(machine_new_keywords_dict))


shenzhen_social_news = get_keywords_of_a_ducment(news_content[4])
print(shenzhen_social_news)


police_mask = np.array(Image.open('/Users/mqgao/Downloads/0034.png_860.png'))
wordcloud_with_mask = wordcloud.WordCloud(
font_path='/Users/mqgao/Downloads/SourceHanSerifSC-Regular.otf', mask=police_mask)
plt.imshow(wc.generate_from_frequencies({w:s for w, s in shenzhen_social_news[:20]}))
plt.imshow(wordcloud_with_mask.generate_from_frequencies({w:s for w, s in shenzhen_social_news[:20]}))
"""