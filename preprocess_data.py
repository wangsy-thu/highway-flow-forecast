import csv
import os

import numpy as np
import pandas as pd

def save_to_csv(dataset_name: str, file_name: str,
                column_list: list, data_list: list):
    """
    csv 文件保存工具方法
    :param dataset_name: 数据集名称
    :param file_name: 文件名称
    :param column_list: 列名称
    :param data_list: 数据列表
    """
    file_path = './data/{}/'.format(dataset_name.upper())

    # 当文件不存在时，创建文件
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    # 打开文件
    with open(file_path + file_name, 'w', encoding='utf-8', newline='') as f:
        # 创建CSV对象
        writer = csv.writer(f)

        writer.writerow(column_list)
        writer.writerows(data_list)


def load_flow_data(dataset_name: str, dtype=np.float32, is_save=True) -> np.ndarray:
    """
    加载流量数据
    :param dataset_name: 数据集名称
    :param dtype: 数据类型
    :param is_save: 是否保存文件
    :return: np.ndarray (T, N, 1)
    """
    flow_df = pd.read_csv('./data/' + dataset_name.upper()
                + '/' + dataset_name.lower() + '_speed.csv')
    flow_mat = np.array(flow_df, dtype=dtype)
    flow_mat = np.expand_dims(flow_mat, axis=2)  # 扩展维度
    if is_save:
        np.savez_compressed(
            './data/' + dataset_name.upper() + '/' + dataset_name.upper() + '.npz',
            data=flow_mat
        )
    return flow_mat



def load_edge_list(dataset_name: str, dtype=np.float32, is_save=True):
    """
    加载边列表数据
    :param dataset_name: 数据集名称
    :param dtype: 数据类型
    :param is_save: 是否保存文件
    :return: np.ndarray (edge_num, 2)
    """
    adj_data = pd.read_csv('./data/' + dataset_name.upper()
                + '/' + dataset_name.lower() + '_adj.csv', header=None)
    adj_mat = np.array(adj_data, dtype=dtype)
    vertices_num, _ = adj_mat.shape
    edge_list = []
    # 该矩阵必须对称

    assert np.all(np.abs(adj_mat - adj_mat.T) < 1e-2)
    for i in range(vertices_num):
        for j in range(i):
            if adj_mat[i, j] > 0:
                edge_list.append([i, j])
    if is_save:
        save_to_csv(dataset_name, '{}.csv'.format(dataset_name.upper()),
                    ['from', 'to'], edge_list)
    return np.array(edge_list, dtype=dtype).T


if __name__ == '__main__':
    flow_dat = load_flow_data('SZ')
    edge_list = load_edge_list('SZ')
    print(edge_list.shape, flow_dat.shape)
    flow_dat = np.load('./data/SZ/SZ.npz')['data']
    print(flow_dat.shape)
