

"""
Why do we need dynamic programming? What's the difference of dynamic programming and previous talked search problme?
Ans:

在很多问题中直接求解会相当困难。我们可以采取分解问题的方法，将复杂问题分解成多个的子问题，这些子问题通常是可以求解的，进而我们可以求解该复杂问题。
搜索问题：找下一步最优
动态规划：将找到的每一步的最优都存起来了

Why do we still need dynamic programming? Why not we train a machine learning to fit a function which could get the right answer based on inputs?
在一些问题中，使用机器学习的方法，由于需要大量的数据，不方便直接求解。使用动态规划来解决可能会更加简单、快捷

Can you catch up at least 3 problems which could solved by Dynamic Programming?
Ans: 编辑距离、硬币找零、背包问题

Can you catch up at least 3 problems wich could sloved by Edit Distance?
Ans: 最小编辑距离、模糊匹配、拼写检查、求最长公共子序列

Please summarize the three main features of Dynamic Programming, and make a concise explain for each feature.
Ans:
问题的具有子问题结构
子问题重复性
对于子问题的解析

What's the disadvantages of Dynamic Programming? (You may need search by yourself in Internet)
Disadvantages：

没有统一的标准模型；
数值方法求解时存在维数灾，消耗空间大，当所给出范围很大时，堆栈中很可能并不能满足所需要的空间大小，
"""

import math
import random
import matplotlib.pylab as plt

latitudes = [random.randint(-100, 100) for _ in range(20)]
longitude = [random.randint(-100, 100) for _ in range(20)]
# plt.scatter(latitudes, longitude)

# 给定一个初始点 $P$, 已经 $k$个车辆，如何从该点出发，经这 k 个车辆经过所以的点全部一次，而且所走过的路程最短?
# 例如：
chosen_p = (5, 10)
plt.scatter(latitudes, longitude)
plt.scatter([chosen_p[0]], [chosen_p[1]], color='r')
plt.show()

def get_distance(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1] -y[1])**2)
# print(distance(chosen_p,(100,100)))

points = [(i,j)for i,j in zip(latitudes,longitude)]
print(points)

sum = 0
for i,v in enumerate(points[:-1]):
    sum += get_distance(v,points[i+1])
print(sum)
sum += get_distance(chosen_p,points[0])
print(sum)

dis_solution = {}
def route_distance(p,points):

    if p in p_set and len(p_set) == 1:return 0
    if len(p_set) == 0:return 0

    candidates = [get_distance(p,e) + route_distance(e,points.remove(e)) for e in points]

    min_distance, point = min(candidates, key=lambda x: x[0])

    dis_solution[(p, points)] = point

    return min_distance

print(chosen_p,points)



