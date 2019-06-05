#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础概念复习
机器学习方法主要用在什么特点的常见下？
Ans: 难以找到规则、模式  同时有大量数据

提出3个你认为使用了机器学习方法的现实场景.
Ans: 商品推荐、银行信贷、天气预报

提出3个你认为可以使用机器学习但是还没有使用机器学习方法的场景.
Ans: 股票预测、疾病诊断、旅游路线推荐

什么是“模型”？ 为什么说“All models are wrong, but some useful”.
Ans: 模型是现实世界的一个抽象。 一般来说，模型对于现实世界的抽象并不是完全正确的，但是在一定程度上对我们要解决的问题有帮助

Classification 和 Regressionu主要针对什么？ 有什么区别？
分类（Classification）是指一类问题，分类的目的在于给对象按照其类别打上相应的标签再分门别类。
回归（regressionu）则是根据样本研究其两个（或多个）变量之间的依存关系，是对于其趋势的一个分析预测。

precision， recall，f1, auc 分别是什么意思？ 假设一个城市有 10000 人，有 30 个犯罪分子，警察抓到了 35 个人，其中 20 个是犯罪分子，请问这个警察的 precision, recall, f1,auc 分别是什么？
Ans: 都是评估一个模型分类好坏程度的指标   precision:20/35 recall:20/25 f1: 2/3 auc:20/10000

请提出两种场景，第一种场景下，对模型的评估很注重 precision, 第二种很注重 recall.
Ans: 在肿瘤判断的情景下，recall相对更重要；
    在嫌疑人定罪情景下，precision更重要；

什么是 Overfitting， 什么是 Underfitting?
Ans:Overfitting:在训练数据和未知数据上表现都很差，高偏差
    Underfitting:在训练数据上表现良好，在未知数据上表现差,高方差

Lazy-Learning， Lazy在哪里？
Ans:
    K-NN是一个Lazy-Learning，因为它不从训练数据中学习判别函数，而是“记忆”训练数据集。

Median， Mode， Mean分别是什么？ 有什么意义？
Ans:
中位数:中位数是一组数据中间位置上的代表值.其特点是不受数据极端值的影响.对于具有偏态分布的数据,中位数的代表性要比均值好(中值能揭示平均值掩盖的真相)
众数:频度最大的那个数
平均数:均值是就全部数据计算的,它具有优良的数学性质,是实际中应用最广泛的集中趋势测度值

Outlinear（异常值、离群值）是什么？ 如何定义？
Ans:
Outlinear:是指在数据中有一个或几个数值与其他数值相比差异较大
定义:发现离群值可以通过观察值的频数表或直方图来初步判断，
如果观测值距（第25百分位数）或（第75百分位数）过远，如两倍以上，则可视该观测值为离群值。

Bias 和 Variance 有什么关系？ 他们之间为什么是一种 tradeoff 的？
Ans:
Bias反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精准度
Variance反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性

Train， Validation，Test 数据集之间是什么关系？ 为什么要这么划分？
Ans:
Train:用来训练模型
Validation:根据验证集上的数据来评估模型表现，对模型进行调整
Test:测试集是数据从来没有见过的数据集，用来测试模型的泛化能力

Supervised Learning 的 Supervised 体现在什么地方？
Ans: 体现在进行学习时，数据需要预先打上标签

Linear Regression 中，什么是“线性关系”？
Ans:数理统计中回归分析，用来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法

Linear Regression中，Loss 函数怎么定义的？ 为什么要写成这样？ 什么是凸函数？ 优化中有什么意义？
Ans: 均方误差  数据的误差越大，增加的惩罚越大  尽可能的找到全部最优点

简述Gradient Descent的过程，以 $y = -10 * x^2 + 3x + 4 $ 为例，从一个任一点 $ x = 10 $ 开始，如果根据 Gradient Descent 找到最值。
Ans:  求导

一般在机器学习数量时，会做一个预处理（Normalization）， 简述 Normalization 的过程，以及数据经过 Normalization之后的平均值和标准差的情况。
Ans: 先求出平均值 再求出方差 根据公式 (x-均值)/方差

Logstic Regression 的 Logstic 是什么曲线，被用在什么地方？
Ans:somgid

Logstic Regression 的 Loss 函数 Cross Entropy 是怎么样的形式？ 有什么意义？
Ans: 是一个值在0-1之间的函数，它将任何数字映射到0-1的区间上，便于用来做分类

"""