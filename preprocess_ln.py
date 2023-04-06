import matplotlib.pyplot as plt
import networkx as nx
import tqdm
from tqdm import trange


def load_highway_grid(data_dir: str, edge_mode: str):
    """
    加载高速路网数据信息
    :param data_dir: 高速节点数据文件名称
    :param edge_mode: 边输出格式 edge_index 与 edge_list
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
    with open(data_dir + '04-辽宁收费站.txt', 'r', encoding='utf-8') as f:
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
    is_reverse = False
    print('=====2-读取门架，构建门架列表=====')
    with open(data_dir + '03-门架.txt', 'r', encoding='utf-8') as f:
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

    if edge_mode == 'edge_index':
        # 重整 edge index 格式
        edge_idx = [[], []]
        for e in edge_index:
            edge_idx[0].append(e[0])
            edge_idx[1].append(e[1])
        return gates, fee_stations, vertex_code_index, edge_idx
    else:
        return gates, fee_stations, vertex_code_index, edge_index


if __name__ == '__main__':
    gate_list, station_list, vertex_code_index, edge_index = load_highway_grid('./data/LN/', edge_mode='edge_list')
    G = nx.Graph()
    G.add_edges_from(edge_index)
    nx.draw_networkx(G, node_size=400, node_color='r')
    plt.show()
