"""
PageRank
"""
import random
import networkx as nx
import matplotlib.pyplot as plt
from string import ascii_uppercase

# ascii_uppercase表示所有的大写字母
print(ascii_uppercase)

def generate_random_website():
    return ''.join([random.choice(ascii_uppercase) for _ in range(random.randint(3,5))]) + '.' + random.choice(['com','cn','net'])

print(generate_random_website())

websites = [generate_random_website() for _ in range(25)] # 随机生成25个域名
print(websites)
print(len(websites))
print(random.sample(websites,10))

websites_connection = {
    websites[0]:random.sample(websites,10),
    websites[1]:random.sample(websites,5),
    websites[2]:random.sample(websites,7),
    websites[3]:random.sample(websites,2),
    websites[4]:random.sample(websites,1)
}

print(websites_connection)
print(websites_connection[websites[0]])
print(websites_connection[websites[1]])

websites_network = nx.graph.Graph(websites_connection)

plt.figure(3,figsize=(12,12))
nx.draw_networkx(websites_network,font_size=10)
plt.show()

sorted(nx.pagerank(websites_network).items(),key=lambda x:x[1],reverse=True)


