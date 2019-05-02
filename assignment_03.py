"""
2. Answer following questions:
1. Why we need machine learning methods instead of creating a complicated formula?
对于人类无法直接编码的过于复杂的任务，需要机器学习。有些任务非常复杂，如果不是不可能的话，人类可以明确地计算出所有的细微差别和代码，这是不切实际的。
因此，我们通过为机器学习算法提供了大量数据，让算法通过探索数据并搜索将实现程序员设定的模型来实现。模型执行的程度由程序员提供的成本函数决定，算法的任务是找到最小化成本函数的模型。

2. Wha't's the disadvantages of the 1st Random Choosen methods in our course?
取值范围有限，且需要多次随机之后，损失函数才可能减小
若出现沿某一方向梯度出现下降，下一次下降仍然是在随机取值，不能继续沿着该方向下降

3. Is the 2nd method supervised direction better than 1st one? What's the disadvantages of the 2nd supversied directin method?
若出现沿某一方向梯度出现下降，下一次下降可以继续沿着该方向下降

4. Why do we use Derivative / Gredient to fit a target function?
一个函数在某一位置的导数，可以表示该函数是递增还是递减
梯度的正方向是函数的递增方向，梯度的反方向是函数递减的方向。

5. In the words 'Gredient Descent', what's the Gredient and what's the Descent?
梯度的正方向是函数f增长最快的方向，梯度的反方向是f降低最快的方向。

6. What's the advantages of the 3rd gradient descent method compared to the previous methods?
增加了学习率，学习率控制每一次下降的步长，在一定程度上可以避免局部最优和跳出当前最优值

7. Using the simple words to describe: What's the machine leanring.
基于数据，拟合出一个函数，使真实值与预测值之间的误差尽可能的小。
"""

import requests
import re
import json
from lxml import etree
import networkx as nx
from collections import defaultdict

start_url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"
}

def get_pages(url):
    response = requests.get(url,headers=headers)

    # html_str = response.content.decode('utf-8')
    html_str = response.content.decode()
    # print(response)
    # print(html_str)
    return html_str

# print(get_pages(start_url))

def get_subway_urls(url):
    urls = []
    html = etree.HTML(get_pages(start_url))
    # /html/body/div[4]/div[2]/div/div[2]/table[1]/tbody/tr[10]/td[2]/div/a
    subway_urls = html.xpath("//div[@class='main-content']/table[1]//div[@class='para']/a/@href")
    for i in subway_urls:
        all = "https://baike.baidu.com" + i
        urls.append(all)
    return urls

# print(get_subway_urls(start_url))
# print(len(get_subway_urls(start_url)))

# 去除重复的subway_urls
def delete_duplicated_subway_urls(list):
    reslut = []
    for e in list:
        if e not in reslut:
            reslut.append(e)
    return reslut

# print(delete_duplicated_subway_urls(get_subway_urls(start_url)))
# print(len(delete_duplicated_subway_urls(get_subway_urls(start_url))))

"""
url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%811%E5%8F%B7%E7%BA%BF'
def get_single_subway_station(url):
    html = etree.HTML(get_pages(url))
    subway_station = html.xpath("//div[@class='main-content']/table[3]//th[1]/text()")
    return subway_station[1:]

print(get_single_subway_station(url))
"""

subway_station_distact = defaultdict(dict)
def get_subway_route_stations_and_distance(subway_usls):
    # subway_usls = delete_duplicated_subway_urls(get_subway_urls(start_url))
    for url in subway_usls:
        html = etree.HTML(get_pages(url))
        route_name = html.xpath("//div[@class='main-content']/table[3]/caption/text()")  # 地铁线路
        subway_station = html.xpath("//div[@class='main-content']/table[3]//th[1]/text()") # 地铁站名
        distance = html.xpath("//div[@class='main-content']/table[3]//tr/td[1]/text()")  # 地铁站与站之间的距离
        distance = re.findall('(\d+)', str(distance))

        temp = defaultdict(dict)
        for k in route_name:
            if subway_station and distance:
                for i,station in enumerate(subway_station[1:]):
            # stations.append(tuple(station.split('——')))
                    temp[tuple(station.split('——'))] = distance[i]
            subway_station_distact[k] = temp

    return subway_station_distact

print(get_subway_route_stations_and_distance(delete_duplicated_subway_urls(get_subway_urls(start_url))))

# 存储爬取的数据
path = 'F:/python/NLP_course/NLP_course/Course_3/beijing_station_distance.json'
with open(path, 'w',encoding='utf-8') as f:
    json_str = json.dumps(str(get_subway_route_stations_and_distance(delete_duplicated_subway_urls(get_subway_urls(start_url)))), indent=4,ensure_ascii=False)
    f.write(json_str)
    f.close()


stations = list(subway_station_distact.keys())
# print(stations)

nodes = []
for k,v in subway_station_distance.items():
    for kk,vv in v.items():
        G.add_edge(kk[0],kk[1])
        if kk[0] not in nodes:
            nodes.append(kk[0])
        if kk[1] not in nodes:
            nodes.append(kk[1])

station_graph = nx.Graph()
station_graph.add_nodes_from(stations)
nx.draw(station_graph,subway_station_distact,with_labels=True,node_size=10)

def is_goal(destination):
    def _wrap(current_path):
        return current_path[-1] == destination
    return _wrap

# print(is_goal(['西直门','大钟寺']))

def sort_path(cmp_func,beam):
    def _sorted(pathes):
        return sorted(pathes,key=cmp_func)[:beam]
    return _sorted

def get_path_distance(path):# Shortest Path Priority（路程最短优先）
    distance = 0
    for i,c in enumerate(path[:-1]):
        distance += get_city_distance(c,path[i+1])

    return distance

def get_total_station(path):# Minimum Transfer Priority(最少换乘优先)
    return len(path)

def get_comprehensive_path(path):#Comprehensive Priority(综合优先)
    return get_path_distance(path) + get_total_station(path)

def get_as_much_path(path):
    return -1 * len(path)

def search(graph,start,is_goal,search_stratgy):
    pathes = [[start] ]
    seen = set()

    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]

        if froniter in seen:continue

        successors = graph[froniter]

        for city in successors:
            if city in path:continue

            new_path = path + [city]

            pathes.append(new_path)

            if is_goal(new_path):return new_path
        # print('len(pathes)={}'.format(pathes))

        seen.add(froniter)
        pathes = search_stratgy(pathes)

# search(cities_connection,start='西直门',is_goal=is_goal('知春路'),search_stratgy=sort_path(get_path_distance,beam=10))
# search(cities_connection,start='西直门',is_goal=is_goal('土桥'),search_stratgy=sort_path(get_total_station))
# search(cities_connection,start='四惠',is_goal=is_goal('土桥'),search_stratgy=sort_path(get_total_station))

"""

4. Create different policies for transfer system.
a. Define different policies for transfer system.

b. Such as Shortest Path Priority（路程最短优先）, Minimum Transfer Priority(最少换乘优先), Comprehensive Priority(综合优先)

c. Implement Continuous transfer. Based on the Agent you implemented, please add this feature: Besides the @param start and @param destination two stations, add some more stations, we called @param by_way, it means, our path should from the start and end, but also include the @param by_way stations.

e.g

1. Input:  start=A,  destination=B, by_way=[C] 
    Output: [A, … .., C, …. B]
2. Input: start=A, destination=B, by_way=[C, D, E]
    Output: [A … C … E … D … B]  
    # based on your policy, the E station could be reached firstly. 
![image.png](attachment:image.png)
"""