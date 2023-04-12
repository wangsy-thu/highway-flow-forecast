import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    edges = pd.read_csv('../../data/LN/LN.csv')
    G = nx.from_pandas_edgelist(edges, source='from', target='to')
    # plot graph
    nx.draw(G, node_size=5, node_color='b', with_labels=False)
    plt.savefig('./plot_feature/networkx_ln.png')
    plt.close()

    # plot node degree distribution
    node_degree = nx.degree(G)
    y_data = [d[1] for d in node_degree]
    degree_val = list()
    degree_count = list()
    plt.title('degree distribution')
    for i in set(y_data):
        degree_count.append(y_data.count(i))
        degree_val.append(i)
    plt.bar(degree_val, degree_count)
    plt.xlabel('node id')
    plt.ylabel('degree')
    plt.savefig('./plot_feature/node_distribution.png')
    plt.close()

    print('=====Graph Map Done=====')
