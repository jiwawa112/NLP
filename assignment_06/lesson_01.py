"""
search Tree -> Similar Words
"""

import re
import jieba
import pandas as pd
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
news_content = [cut(n) for n in news_content if n != 'n']
print(news_content[0])

with open('news-sentences-cut.txt', 'w') as f:
    for n in news_content:
        f.write(n + '\n')

news_word2Vec = Word2Vec(LineSentence('news-sentences-cut.txt'),size=35,workers=8)
print(news_word2Vec.most_similar('葡萄牙'))
print(news_word2Vec.most_similar('葡萄牙',topn=20))
print(news_word2Vec.most_similar('捷克',topn=20))
print(news_word2Vec.most_similar('说',topn=30))
print(news_word2Vec.most_similar('认为',topn=30))
print(news_word2Vec.most_similar('建议',topn=30))
# More Date,Better Results
# (1) 分词的问题
# (2) 数据量，数据越多，效果越好，维基百科加进来，那么同义词就要好很多

def get_related_words(initial_words,model):
    """
    @initial_words are initial words we already konw
    @model is the word2vec model
    """
    unseen = initial_words

    seen = defaultdict(int)

    max_size = 500 # could be bigger

    while unseen and len(seen) < max_size:
        if len(seen) % 50 == 0:
            print('seen length:{}'.format(len(seen)))
        node = unseen.pop(0)

        new_expanding = [w for w,s in model.most_similar(node,topn=20)]

        unseen += new_expanding

        seen[node] += 1

        # optimal:1. score function could be revised
        # optimal:2. using dymanic programming to reduce computing time

    return seen
print(len(news_word2Vec.wv.vocab))
# print(get_related_words(['说','表示'],news_word2Vec))
related_words = get_related_words(['说','表示',news_word2Vec])

sorted(related_words.items(),key=lambda x:x[1],reverse=True)