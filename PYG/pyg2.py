import networkx as nx

# 创建一个无向图
G = nx.Graph()

# 添加节点
G.add_node(1)
G.add_node(2)



# 添加边
G.add_edge(0, 1)
G.add_edge(1, 0)
G.add_edge(1, 2)
G.add_edge(2, 1)

# 打印图的节点和边
print("Nodes:", G.nodes())
print("Edges:", G.edges())

# 可视化图
import matplotlib.pyplot as plt
nx.draw(G, with_labels=True)
plt.show()