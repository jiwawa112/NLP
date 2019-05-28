# TFIDF Vectoized
import random
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
import pandas as pd

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
news_content = [cut(n) for n in news_content if n != 'n']
print(news_content[0])

vectoized = TfidfVectorizer(max_features=10000)
print(news_content[:10])

sample_num = 1000
sub_samples = news_content[:sample_num]

X = vectoized.fit_transform(sub_samples)
print(X.shape)
print(vectoized.vocabulary)

np.where(X[0].toarray()) # get the positions which values are not zero

document_id_1, document_id_2 = random.randint(0, 1000), random.randint(0, 1000)
print(document_id_1)
print(document_id_2)
print(news_content[document_id_1])
print(news_content[document_id_2])

vector_of_d_1 = X[document_id_1].toarray()[0]
vector_of_d_2 = X[document_id_2].toarray()[0]
random_choose = random.randint(0, 1000)
print(random_choose)
print(news_content[random_choose])

def distance(v1, v2): return cosine(v1, v2)
print(distance([1, 1], [2, 2]))
distance(X[random_choose].toarray()[0], X[document_id_1].toarray()[0])
distance(X[random_choose].toarray()[0], X[document_id_2].toarray()[0])
print(news_content[320])
print(news_content[72])
print(news_content[85])
print(news_content[8])
sorted(list(range(10000)), key=lambda i: distance(X[random_choose].toarray()[0],
                                      X[i].toarray()[0]))
bin(19)
bin(49)
bin(38)
bin(49 & 38)