import csv
import json
import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
from tqdm import trange


def load_highway_grid(data_dir: str, edge_mode: str, file_encoding='utf-8'):
    """
    加载高速路网数据信息
    :param data_dir: 高速节点数据文件名称
    :param edge_mode: 边输出格式 edge_index 与 edge_list
    :param file_encoding: 读取文件编码格式
    :return: gate_list, station_list, vertices_index, edge_index
            vertices_index: {
                vertex_name: vertex_id
            }
            edge_index: list[list], shape: (edge_nums, 2)
    """

    # 1, 加载高速收费站数据
    fee_stations = []
    count = 0
    print('=====1-读取收费站，构建收费站列表=====')
    with open(data_dir + '04-辽宁收费站.txt', 'r', encoding=file_encoding) as f:
        f.readline()  # 从第二行开始读
        rows = f.readlines()
        for row in tqdm.tqdm(rows, desc='读取收费站数据'):
            fee_station_info = row.split(',')
            fee_station_code = fee_station_info[0]  # 收费站编号
            fee_station_name = fee_station_info[1]  # 收费站名称
            fee_station_post_code = fee_station_info[2]  # 桩号
            fee_station_longitude = fee_station_info[3]  # 经度
            fee_station_latitude = fee_station_info[4]  # 纬度

            # 抽取列表，加入数据库
            fee_stations.append(
                {
                    'id': count,
                    'code': fee_station_code,
                    'name': fee_station_name,
                    'post_code': fee_station_post_code,
                    'longitude': fee_station_longitude,
                    'latitude': fee_station_latitude[:-1]
                }
            )
            count += 1

    # 2, 加载门架数据
    gates = []
    bridge_vertex_names = []  # 虚拟节点信息
    reverse_count = 0
    station_count = count
    is_reverse = False
    print('=====2-读取门架，构建门架列表=====')
    with open(data_dir + '03-门架.txt', 'r', encoding=file_encoding) as f:
        f.readline()  # 从第二行开始读
        rows = f.readlines()
        for row in tqdm.tqdm(rows, desc='读取门架数据'):
            gate_info = row.split(',')
            gate_code = gate_info[0]  # 门架编号
            gate_name = gate_info[1]  # 门架名称
            gate_start = gate_info[1].split('-')[0]  # 门架所在路段起点
            gate_end = gate_info[1].split('-')[1][:-1]  # 门架所在路段终点
            gate_post_code = gate_info[2]  # 门架桩号
            gate_longitude = gate_info[3]  # 门架经度
            gate_latitude = gate_info[4]  # 门架纬度
            gate_label = (gate_name[-1] == '1')  # 门架是否为第一个

            if (not is_reverse) and ((reverse_count == 0 and gate_label) or (not gate_label)):
                # 加入门架列表 且 当前为正向
                gates.append(
                    {
                        'id': count,
                        'code': gate_code,
                        'name': gate_name,
                        'start': gate_start,
                        'end': gate_end,
                        'post_code': gate_post_code,
                        'longitude': gate_longitude,
                        'latitude': gate_latitude[:-1],
                    }
                )
                # 加入桥接节点列表
                if '收费站' not in gate_start:
                    bridge_vertex_names.append(
                        gate_start
                    )
                if '收费站' not in gate_end:
                    bridge_vertex_names.append(
                        gate_end
                    )
                reverse_count += 1
                count += 1
                is_reverse = False
            elif reverse_count > 0 and gate_label:
                # 反向门架序列
                is_reverse=True
                gates.append(
                    {
                        'id': count - reverse_count,
                        'code': gate_code,
                        'name': gate_name,
                        'start': gate_start,
                        'end': gate_end,
                        'post_code': gate_post_code,
                        'longitude': gate_longitude,
                        'latitude': gate_latitude[:-1],
                    }
                )
                reverse_count -= 1  # 一个反向完成
                if reverse_count == 0:
                    is_reverse = False
            elif reverse_count > 0 and is_reverse:
                gates.append(
                    {
                        'id': count - reverse_count,
                        'code': gate_code,
                        'name': gate_name,
                        'start': gate_start,
                        'end': gate_end,
                        'post_code': gate_post_code,
                        'longitude': gate_longitude,
                        'latitude': gate_latitude[:-1],
                    }
                )
                reverse_count -= 1  # 一个反向完成
                if reverse_count == 0:
                    is_reverse = False

    # 3, 构建高速路网节点索引
    vertex_name_index = {}  # 名称索引
    vertex_code_index = {}  # 编号索引
    print('=====3-构建节点索引=====')
    for fee_station in tqdm.tqdm(fee_stations, desc='收费站索引'):
        vertex_name_index[fee_station['name'][2: -1] + '收费站'] = fee_station['id']
        vertex_code_index[fee_station['code']] = fee_station['id']

    # 4, 构建门架索引
    for gate in tqdm.tqdm(gates, desc='门架索引'):
        vertex_code_index[gate['code']] = gate['id']
        vertex_name_index[gate['name']] = gate['id']

    # 5, 构建桥接节点索引
    vertices_num = count
    for bridge_vertex in tqdm.tqdm(bridge_vertex_names, desc='桥接点索引'):
        vertex_name_index[bridge_vertex] = count
        count += 1

    print('=====Vertex Count: {}, Total Count: {}====='.format(
        vertices_num, count
    ))

    print('=====4-构建 Edge Index=====')

    # 6, 构建高速路网邻接矩阵
    edge_indices = []
    bridge_vertex_index = {}  # 桥接节点索引，key为node id，value 为 Gate or Station ID
    for v_idx in range(vertices_num, count):
        bridge_vertex_index[v_idx] = [], []
    previous_gate_idx = 0
    reverse_count = 0
    idx = 0
    end_idx = 0
    edge_count = 0
    while idx < len(gates):
        gate = gates[idx]
        if gate['name'][-1] == '1' and reverse_count == 0:
            # 第一种情况 (节点) -> (门架) -> (节点)
            start_idx = vertex_name_index[gate['start']]
            gate_idx = vertex_name_index[gate['name']]
            end_idx = vertex_name_index[gate['end']]
            edge_indices.append([start_idx, gate_idx])
            edge_count += 1

            if start_idx >= vertices_num:
                # 桥接节点
                bridge_vertex_index[start_idx][0].append(gate_idx)
                bridge_vertex_index[start_idx][1].append(edge_count - 1)

            edge_indices.append([gate_idx, end_idx])
            edge_count += 1

            previous_gate_idx = gate_idx
            reverse_count += 1
            idx += 1
        elif gate['name'][-1] != '1':
            # 第二种情况 (节点) -> (门架) -> (门架) -> (节点)
            gate_idx = vertex_name_index[gate['name']]
            end_idx = vertex_name_index[gate['end']]

            # 01-消除 previous_gate_idx 与 end_idx 之间的连接
            edge_indices.pop()

            # 02-建立 previous_gate_idx 与 gate_idx, gate_idx 与 end_idx 之间连接
            edge_indices.append([previous_gate_idx, gate_idx])
            edge_indices.append([end_idx, gate_idx])
            edge_count += 1

            previous_gate_idx = gate_idx
            reverse_count += 1
            idx += 1

        elif gate['name'][-1] == '1' and reverse_count > 0:

            if end_idx >= vertices_num:
                # 桥接节点
                bridge_vertex_index[end_idx][0].append(previous_gate_idx)
                bridge_vertex_index[end_idx][1].append(edge_count - 1)

            # 开始反向操作，跳过 reverse_count 个门架即可
            idx += reverse_count
            reverse_count = 0

    # 6, 删除虚拟节点
    del_edge_index = []
    new_edges = []
    print('=====5-删除Bridge Vertices, 重整=====')
    for del_idx in trange(vertices_num, count):
        gate_idx_lst, edge_idx_lst = bridge_vertex_index[del_idx]
        for i in range(len(gate_idx_lst) - 1):
            for j in range(i + 1, len(gate_idx_lst)):
                new_edges.append([
                    gate_idx_lst[i],
                    gate_idx_lst[j]
                ])
        del_edge_index.extend(edge_idx_lst)

    edge_index = [edge for idx, edge in enumerate(edge_indices) if idx not in del_edge_index]
    edge_index.extend(new_edges)

    # 7, 保存到文件
    edge_idx_column_list = [
        'from',
        'to'
    ]
    save_to_csv('LN.csv', edge_idx_column_list, edge_index)

    vertex_index_file_path = './data/LN/index/'
    shutil.rmtree(vertex_index_file_path)
    if not os.path.isfile(vertex_index_file_path):
        os.mkdir(vertex_index_file_path)

    with open(vertex_index_file_path + 'vertex_index.json', 'w') as fj:
        json.dump({
            'vertices_num': vertices_num,
            'station_num': station_count,
            'gate_num': vertices_num - station_count,
            'vertex_index': vertex_code_index
        }, fj, indent=2)

    if edge_mode == 'edge_index':
        # 重整 edge index 格式
        edge_idx = [[], []]
        for e in edge_index:
            edge_idx[0].append(e[0])
            edge_idx[1].append(e[1])
        return gates, fee_stations, vertex_code_index, edge_idx
    else:
        return gates, fee_stations, vertex_code_index, edge_index


def save_to_csv(file_name: str, column_list: list, data_list: list):
    """
    csv 文件保存工具方法
    :param file_name: 文件名称
    :param column_list: 列名称
    :param data_list: 数据列表
    """
    file_path = './data/LN/'

    # 当文件不存在时，创建文件
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    # 打开文件
    with open(file_path + file_name, 'w', encoding='utf-8', newline='') as f:

        # 创建CSV对象
        writer = csv.writer(f)

        writer.writerow(column_list)
        writer.writerows(data_list)



def get_time_step_idx(flow_time: str, flow_period: str) -> int:
    """
    获取时间步索引工具方法
    :param flow_time: 流量产生时间
    :param flow_period: 流量统计周期
    :return: 流量所在时间步 ID
    """
    if flow_period == 'daily':
        return int(flow_time[2: 4]) * 12 + int(flow_time[4: 6]) // 5
    elif flow_period == 'month':
        return (int(flow_time[:2]) - 1)* 24 * 12 \
            + int(flow_time[2: 4]) * 12 \
            + int(flow_time[4: 6]) // 5
    else:
        return int(flow_time[2: 4]) * 12 + int(flow_time[4: 6]) // 5


def load_unit_highway_flow(flow_data_dir: str, flow_period: str,
                           period_length: int, vertex_index: dict,
                           vertices_num: int, offset: int, batch_size=20) -> np.ndarray:
    """
    生成每个时间单位的高速流量数据，上层调用读取每日数据后进行 concatenate 操作
    :param flow_data_dir: 流量数据所在文件夹
    :param vertices_num: 高速路网节点数量
    :param flow_period: 流量计算周期，为 daily 与 month
    :param period_length: 每个周期时间步数量
    :param vertex_index: 节点代码索引
    :param offset: 偏移量
    :param batch_size: 批量读取大小，减少 IO 次数
    :return: np.ndarray (
        vertices_num, feature_num, time_step_num
    )
    """

    flow_data_file_list = [
        flow_data_dir + d for d in os.listdir(flow_data_dir)
    ]

    # 数据格式 (time_step_num, vertices_num, feature_num)
    # 特征分别为 (
    #       0: 车流量总量
    #       1: 平均收费
    #       2: 1型号车辆数量 (客车)
    #       3: 2型号车辆数量 (货车)
    #       4: 3型号车辆数量 (限行)
    #       5: 0型号车型数量 (未知)
    # )
    all_flow = []

    for flow_file_name in tqdm.tqdm(flow_data_file_list, desc='整理月份流量'):
        daily_flow_mat = np.zeros(
            shape=(period_length, vertices_num, 6)
        )
        with open(flow_file_name, 'r', encoding='utf-8') as f:
            while True:
                rows = f.readlines(batch_size)

                # 文件读取完成
                if len(rows) == 0:
                    break

                # 循环读取流量数据
                for row in rows:
                    flow_item_info = row.split(',')
                    vertex_id = vertex_index.get(flow_item_info[0])

                    if vertex_id is None:
                        # 若流量中节点不存在，跳过，继续读取
                        continue
                    vertex_id -= offset  # 门架号(收费站号)

                    time_step_id = get_time_step_idx(flow_item_info[1][6:], flow_period)  # 时间步 ID
                    flow_fee = int(flow_item_info[2])  # 产生费用 (cent)
                    vehicle_type = int(flow_item_info[3])  # 车型

                    if 0 < vehicle_type < 10:
                        vehicle_type_id = 2
                    elif 10 <= vehicle_type < 20:
                        vehicle_type_id = 3
                    elif vehicle_type == 9:
                        vehicle_type_id = 4
                    else:
                        vehicle_type_id = 5

                    daily_flow_mat[time_step_id, vertex_id, 0] += 1  # 流量
                    daily_flow_mat[time_step_id, vertex_id, 1] += (flow_fee / 100)  # 金额
                    daily_flow_mat[time_step_id, vertex_id, vehicle_type_id] += 1  # 各种车型
        all_flow.append(daily_flow_mat.copy())
    return np.concatenate(all_flow, axis=0)


def load_highway_flow(vertex_index_dir: str, data_dir: str, save_dir: str,
                      is_save: bool, day_count_list: list) -> np.ndarray:
    """
    生成高速流量数据并保存
    :param vertex_index_dir: 索引文件地址
    :param data_dir: 数据的文件夹
    :param save_dir: 保存的文件夹
    :param is_save: 是否将流量矩阵保存到文件
    :param day_count_list: 每月天数列表
    :return: 流量矩阵
    """

    # 加载路网节点索引
    print('=====开始流量整理=====')
    with open(vertex_index_dir + 'vertex_index.json', 'r') as jf:
        j = json.load(jf)
        vertex_index = j['vertex_index']
        gate_num = j['gate_num']
        station_num = j['station_num']

    # 1, 整理门架流量
    print('=====1, 整理门架流量=====')
    month_gate_dirs = os.listdir(data_dir + 'gate/')
    gate_flow_list = []

    for month_id, month_dir in enumerate(month_gate_dirs):
        print('=====Processing Gate Flow: Month {}====='.format(month_id))
        month_flow_mat = load_unit_highway_flow(
            flow_data_dir=data_dir + 'gate/' + month_dir + '/',
            flow_period='daily',
            period_length=24 * 12,
            vertex_index=vertex_index,
            vertices_num=gate_num,
            offset=station_num,
            batch_size=20
        )
        gate_flow_list.append(month_flow_mat.copy())
    all_gate_flow_mat = np.concatenate(gate_flow_list, axis=0)

    # 2, 整理收费站流量
    month_station_dirs = os.listdir(data_dir + 'station/')
    station_flow_list = []

    print('=====2, 整理收费站流量=====')
    for month_id, month_dir in enumerate(month_station_dirs):
        print('=====Processing Station Flow: Month {}====='.format(month_id))
        month_flow_mat = load_unit_highway_flow(
            flow_data_dir=data_dir + 'station/' + month_dir + '/',
            flow_period='month',
            period_length=day_count_list[month_id] * 24 * 12,
            vertex_index=vertex_index,
            vertices_num=station_num,
            offset=0,
            batch_size=20
        )
        station_flow_list.append(month_flow_mat.copy())
    all_station_flow_mat = np.concatenate(station_flow_list, axis=0)

    # 3, 合并流量
    all_flow_mat = np.concatenate([all_station_flow_mat, all_gate_flow_mat], axis=1)

    if is_save:
        np.savez_compressed(
            save_dir + 'LN.npz',
            data=all_flow_mat
        )

    return all_flow_mat


if __name__ == '__main__':
    gate_list, station_list, vertex_code_index, edge_index = load_highway_grid('./data/LN/', edge_mode='edge_list')

    # 可视化路网结构
    G = nx.Graph()
    G.add_edges_from(edge_index)
    nx.draw_networkx(G, node_size=5, node_color='b', with_labels=False)
    plt.show()

    flow_data_mat = load_highway_flow(
        vertex_index_dir='./data/LN/index/',
        data_dir='./data/LN/flow/',
        save_dir='./data/LN/',
        is_save=True,
        day_count_list=[
            2, 2
        ]
    )

    print(flow_data_mat[0, :, :])
    npz_data = np.load('./data/LN/LN.npz')
    flow_mat = npz_data['data']
    print(flow_mat.shape)
