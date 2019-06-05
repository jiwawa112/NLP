#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

numbers = [random.randint(0,100) for i in range(100)]
print(numbers)

# Mean 平均数
print(np.mean(numbers))

# Module 众数
# 是一组数据中出现次数最多的数值，众数在一组数中可能有好几个。用M表示
# 简单的说，就是一组数据中占比例最多的那个数
counts = np.bincount(numbers) # 统计非负整数的个数
print(np.argmax(counts)) # 返回众数

# 中位数
# 中位数是按顺序排列的一组数据中居于中间位置的数，即在这组数据中，有一半的数据比他大，有一半的数据比他小
print(np.median(numbers))

# 标准差、方差
numbers = [random.randint(0,100) for i in range(100)]
std = np.std(numbers)
mean = np.mean(numbers)

# 数据的Normalization(数据的缩放)
print(np.std([(n - mean) / std for n in numbers]))
print(np.mean([(n - mean) /std for n in numbers]))

# Outline 异常值
np.percentile(numbers,25) / 1.5
np.percentile(numbers,75) / 1.5

s = [n for n in numbers if n < np.percentile(numbers,25) / 1.5]
print(s)

