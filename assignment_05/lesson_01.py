from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

import jieba
import pandas as pd

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences,min_count=1)

print(model.wv.vocab)
print(model.most_similar('cat'))
print(model.wv['dog'])

# data_path = 'F:\python\NLP_course\NLP_course\Course_5\sqlResult_1558435.csv'
content = pd.read_csv('sqlResult_1558435.csv',encoding='gb18030')
samples = content['content'][:100]
print(samples)

def cut(string):return ' '.join(jieba.cut(string))

with open('mini_samples.txt','w',encoding='utf-8') as f:
    for s in samples:
        f.write(cut(s)+'\n')
sentences = LineSentence('mini_samples.txt')
model = Word2Vec(sentences, min_count=1)
print(model.wv['小米'])
print(model.wv.most_similar('科学家'))
print(model.wv.vocab)

# Name Entity Recognition
text = """
新华社华盛顿4月26日电 美国总统特朗普26日表示，美国将撤销在《武器贸易条约》上的签字。
特朗普当天在美国印第安纳州首府印第安纳波利斯举行的美国全国步枪协会年会上说，
《武器贸易条约》是一个“严重误导的条约”，美国将撤销在该条约上的签字，联合国将很快收到美国正式拒绝该条约的通知。
"""

for w in posseg.cut(text):
    print(w)
