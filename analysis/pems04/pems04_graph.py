import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    edges = pd.read_csv('../../data/PEMS04/PEMS04.csv')
    G = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr='cost')
    # plot graph
    nx.draw(G, node_size=30, node_color='r')
    plt.savefig('./plot_feature/networkx_pems04.png')
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

    # plot page-rank
    data_pg = nx.pagerank(G)
    rank_lst = data_pg.values()
    plt.title('page rank')
    plt.bar([i for i in range(len(rank_lst))], rank_lst)
    plt.xlabel('node')
    plt.ylabel('page rank')
    plt.savefig('./plot_feature/page_rank.png')
    plt.figure()
