"""
2. Review the main points of this lesson.
1. How to Github and Why do we use Jupyter and Pycharm;
Git是一个分布式版本控制系统


2. What's the Probability Model?
概率模型是描述了一个或多个随机变量的之间的概率事件的集合,通过概率集合组合成的数学模型叫做概率模型

3. Can you came up with some sceneraies at which we could use Probability Model?
天气预报、交通流量、

4. Why do we use probability and what's the difficult points for programming based on parsing and pattern match?
语言是相当复杂的，基于人工撰写的规则，这种方法费时费力，且不能覆盖各种各样的语言现象
比如说自然语言中词的一词多意，难以用规则来描述，它是依赖于上下文来理解的。


5. What's the Language Model;
简单地说，语言模型就是用来计算一个句子出现的概率的模型，即P(W1,W2,...Wk)。
判断一个句子是否合理，看它的可能性大小即可。
利用语言模型，可以确定哪一个词序列出现的可能性更大，或者给定若干个词，可以预测下一个最可能出现的词语。

6. Can you came up with some sceneraies at which we could use Language Model?
机器翻译、语言识别、对话机器人、文本分类、情感分析

7. What's the 1-gram language model;
单个词在语料库中出现的概率，通过将句子中的每一个词出现的概率连乘，得出一个句子出现的概率

8. What's the disadvantages and advantages of 1-gram language model;
优势：简单
缺点：通过单个词的出现概率，无需判断词语的先后顺序。

9. What't the 2-gram models;
一次考虑两个词的出现概率，如在词A出现的情况下词B出现的概率。考虑了词与词之间出现顺序的情况

10. what's the web crawler, and can you implement a simple crawler?
网络爬虫，是一种按照一定的规则，自动地抓取万维网信息的程序或者脚本

11. There may be some issues to make our crwaler programming difficult, what are these, and how do we solve them?
1.爬取的数据量非常大    通过分布式爬虫采集
2.网站一般会有一些反爬虫的机制     争对特定的机制采取不用的策略 如：降低爬取速率、IP代理、登陆、验证码等

12. What't the Regular Expression and how to use?
通过一定的规则对文本中内容进行检索、提取、替换等操作，帮助提取文本中想要获得的内容。
re库
"""

import re
import glob
from hanziconv import HanziConv
import jieba
import numpy as np
import pandas as pd
from collections import Counter
from functools import reduce
import matplotlib.pyplot as plt

file_path = 'F:/python/NLP_course/NLP_course/Course_2/extracted/**/wiki_**'
files = glob.glob(file_path)

all_articles = []

for file in files:
    text = open(file,encoding='utf8')
    lines = text.readlines()
    all_articles.append(lines)
print(all_articles[0])
# print(all_articles[1])
# print(all_articles[:5])
# print(type(all_articles[0]))

# 过滤文本中的<>
pat = re.compile('<[^>]+>')
TEXT = pat.sub('',str(all_articles))

# 转换为中文简体
TEXT = HanziConv.toSimplified(TEXT)
# print(TEXT)

def tokens(string):
    return ' '.join(re.findall('[\w|\d]+', string))

TEXT_TOKENS = tokens(TEXT)
print(TEXT_TOKENS[0])

# 得到分词
def cut(string):
    return list(jieba.cut(string))

ALL_TOKENS = cut(TEXT_TOKENS)


valid_tokens = [t for t in ALL_TOKENS if t.strip()]
print(len(ALL_TOKENS))
print(len(valid_tokens))

# Get the frequences of words
words_count = Counter(valid_tokens)
words_count.most_common(10) # 出现频率最高的10个词
print(words_count.most_common(10))

frequences = [f for w, f in words_count.most_common(100)]
x = [i for i in range(len(frequences[:100]))]
print(len(frequences))

# 查看前100个词的词频
plt.plot(x,frequences)
plt.plot(x,np.log(frequences))

frequences_all = [f for w,f in words_count.most_common()]
frequences_sum = sum(frequences_all)
print(frequences_sum)


def get_prob(word):
    eps = 1 / frequences_sum
    if word in words_count:
        return words_count[word] / frequences_sum
    else:
        return eps

print(get_prob('飞机'))
print(get_prob('我们'))

def product(numbers):
    return reduce(lambda n1,n2:n1 *n2,numbers)

# one-gram-model
def language_model_one_gram(string):
    words = cut(string)
    return product([get_prob(w) for w in words])


print(language_model_one_gram('成都糖酒会下个月开幕'))
print(language_model_one_gram('小明毕业于清华大学'))

sentences = """
这是一个比较正常的句子
这个一个比较罕见的句子
小明毕业于清华大学
小明毕业于秦华大学
""".split()

for s in sentences:
    print(s,language_model_one_gram(s))

need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "我去吃火锅，今晚 今晚我去吃火锅"
]

for s in need_compared:
    s1,s2 = s.split()
    p1,p2 = language_model_one_gram(s1), language_model_one_gram(s2)

    better = s1 if p1 > p2 else s2

    print('{} is more possible'.format(better))
    print('-' * 4 + ' {} with probility {}'.format(s1, p1))
    print('-' * 4 + ' {} with probility {}'.format(s2, p2))

# 2-Gram-model
# Get the combination probability of 2-grams
# 每次考虑两个词的概率
valid_tokens = [str(t) for t in valid_tokens]
all_2_grams_words = [''.join(valid_tokens[i:i+2]) for i in range(len(valid_tokens[:-2]))]

_2_gram_sum = len(all_2_grams_words)
_2_gram_counter = Counter(all_2_grams_words)

def get_combination_prob(w1, w2):
    if w1 + w2 in _2_gram_counter: return _2_gram_counter[w1+w2] / _2_gram_sum
    else:
        return 1 / _2_gram_sum

# 表示'去'和'北京'两个词同时出现的概率
print(get_combination_prob('去','北京'))
print(get_combination_prob('波音', '飞机'))
print(get_combination_prob('苹果', '手机'))

def get_prob_2_gram(w1, w2):
    return get_combination_prob(w1, w2) / get_prob(w1)

# 已知第一个词时'去'、第二个词时'沈阳'的概率
print(get_prob_2_gram('去', '沈阳'))
print(get_prob_2_gram('去', '北京'))

def langauge_model_of_2_gram(sentence):
    sentence_probability = 1

    words = cut(sentence)

    for i, word in enumerate(words):
        if i == 0:
            prob = get_prob(word)
        else:
            previous = words[i - 1]
            prob = get_prob_2_gram(previous, word)
        sentence_probability *= prob

    return sentence_probability

print(langauge_model_of_2_gram('小明今天奖天抽到一台苹果手机'))
print(langauge_model_of_2_gram('小明今天抽奖抽到一台波音飞机'))

# Review the problem using 2-gram
need_compared = [
    "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
]

for s in need_compared:
    s1, s2 = s.split()
    p1, p2 = langauge_model_of_2_gram(s1), langauge_model_of_2_gram(s2)

    better = s1 if p1 > p2 else s2

    print('{} is more possible'.format(better))
    print('-' * 4 + ' {} with probility {}'.format(s1, p1))
    print('-' * 4 + ' {} with probility {}'.format(s2, p2))

