#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dependency Parsing 依存分析
import os
from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import SentenceSplitter
from pyltp import NamedEntityRecognizer
from pyltp import SementicRoleLabeller

sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！') # 分句
print('\n'.join(sents))


LTP_DATA_DIR = 'F:/python/NLP_course/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl_win.model')

# 分词
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
words = segmentor.segment('元芳你怎么看？')  # 分词
print('\t'.join(words))
segmentor.release()  # 释放模型

# 词性标注
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型
words = ['元芳','你','怎么','看'] # 分词结果
postags = postagger.postag(words)  # 词性标注
print('\t'.join(postags))
postagger.release()  # 释放模型

# 命名实体识别
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型
words = ['元芳','你','怎么','看']
postags = ['nh','r','r','v']
netags = recognizer.recognize(words, postags)  # 命名实体识别

print('\t'.join(netags))
recognizer.release()  # 释放模型

"""
LTP 采用 BIESO 标注体系。B 表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体，O表示不构成命名实体。
LTP 提供的命名实体类型为:人名（Nh）、地名（Ns）、机构名（Ni）。
B、I、E、S位置标签和实体类型标签之间用一个横线 - 相连；O标签后没有类型标签。
"""

# 依存句法分析
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型
words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
arcs = parser.parse(words, postags)  # 句法分析
print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
parser.release()  # 释放模型

# 语义角色标注
labeller = SementicRoleLabeller() # 初始化实例
labeller.load(srl_model_path)  # 加载模型
words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
# arcs 使用依存句法分析的结果
roles = labeller.label(words, postags, arcs)  # 语义角色标注

# 打印结果
for role in roles:
    print(role.index, "".join(
        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
labeller.release()  # 释放模型

length = len(words)
print(len(words),len(postags),len(netags))
for i in range(length):
    print('{} {} {} '.format(words[i],postags[i],netags[i]))