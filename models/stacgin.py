import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from utils.adj import get_edge_index
from utils.statistics import get_parameter_number


class SpatialAttentionLayer(nn.Module):
    """
    Spatial Attention Layer
    空间注意力层，计算 N 个节点的注意力矩阵
    """

    def __init__(self, in_channels: int, vertices_num: int, time_steps_num: int):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param time_steps_num: 时间步数量，指的是输入的时间段的个数， Integer 类型
        """
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(time_steps_num))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, time_steps_num))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, vertices_num, vertices_num))
        self.Vs = nn.Parameter(torch.FloatTensor(vertices_num, vertices_num))

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, F_in, T)
        :return: 输出为空间注意力矩阵 (B, N, N)
        """
        # 消除特征维度 F
        mat_n_l = torch.matmul(torch.matmul(x_matrix, self.W1), self.W2)  # (B, N, T)
        # 另外一半消除特征维度 F
        mat_n_r = torch.matmul(self.W3, x_matrix).transpose(-1, -2)  # (B, T, N)
        dot_product = torch.matmul(mat_n_l, mat_n_r)  # (B, N, N)
        # 对 dot_product Attention 做线性重组

        S_att = torch.matmul(self.Vs, torch.sigmoid(dot_product + self.bs))
        # 做 Softmax 归一化，使得注意力输出每行和为 1
        S_norm = F.softmax(S_att, dim=1)
        return S_norm


class TemporalAttentionLayer(nn.Module):
    """
    时间注意力机层，计算所有时间步的注意力矩阵
    """

    def __init__(self, in_channels: int, vertices_num: int, time_steps_num: int):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param time_steps_num: 时间步数量，指的是输入的时间段的个数， Integer 类型
        """
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(vertices_num))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, vertices_num))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, time_steps_num, time_steps_num))
        self.Ve = nn.Parameter(torch.FloatTensor(time_steps_num, time_steps_num))

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, C, T)
        :return: 输出为时间注意力矩阵 (B, T, T)
        """
        # 消除特征维度 F (B, T, N)
        mat_t_l = torch.matmul(torch.matmul(x_matrix.permute(0, 3, 2, 1), self.U1), self.U2)
        # 右侧消除特征维度 F (B, N, T)
        mat_t_r = torch.matmul(self.U3, x_matrix)
        dot_product = torch.matmul(mat_t_l, mat_t_r)  # (B, T, T)
        # 线性重分布
        E_att = torch.matmul(self.Ve, torch.sigmoid(dot_product + self.be))  # (B, T, T)
        E_norm = F.softmax(E_att, dim=1)
        return E_norm

class GinConvLayer(nn.Module):
    """
    基于 GAT 卷积的空间卷积模型
    """
    def __init__(self, in_channels: int, out_channels: int,
                 edge_index: torch.Tensor, device, K=2):
        """
        构造函数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param device: 模型所在设备
        """
        super(GinConvLayer, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        # 多层 GAT 卷积
        self.ginConvList = nn.ModuleList([
            GINConv(
                nn=nn.Linear(
                    in_features=in_channels,
                    out_features=out_channels
                )
            )
        ])
        self.ginConvList.extend([
            GINConv(
                nn=nn.Linear(
                    in_features=out_channels,
                    out_features=out_channels
                )
            ) for _ in range(self.K - 1)
        ])
        self.edge_index = edge_index

    def forward(self, graph_sig):
        """
        正向传播函数
        :param graph_sig: 图信号，这里是一个时间步的数据 (B, N, C)
        :return: 图卷积后的结果 (B, N, C_out)
        """
        for ginLayer in self.ginConvList:
            graph_sig = ginLayer(graph_sig, self.edge_index)
        return graph_sig


class SpatialConvLayer(nn.Module):
    """
    Spatial Convolution Layer
    多层 GIN 空间卷积层
    """

    def __init__(self, K: int, edge_index: torch.Tensor,
                 in_channels: int, out_channels: int, device):
        """
        构造函数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param device: 模型所在设备
        """
        super(SpatialConvLayer, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.ginConv = GinConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_index=edge_index,
            device=device,
            K=K
        )

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入矩阵数据 (B, N, F_in, T)
        :return: 输出矩阵数据 (B, N, F_out, T)
        """

        batch_size, vertices_num, in_channel, time_step_num = x_matrix.size()
        outputs = []

        # 对每一个时间步，整合空间注意力做空间卷积
        for time_step in range(time_step_num):
            # 提取该时间步下的图信号
            graph_sig = x_matrix[:, :, :, time_step]
            # 自定义的中间变量，不参与模型的梯度更新，
            output = self.ginConv(graph_sig)
            # 堆叠输出
            outputs.append(output.unsqueeze(-1))  # (B, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))


class STACGINBlock(nn.Module):
    """
    STACGIN Block
    这里定义一个 STACGIN 块，后续可以通过块堆叠增加模型表达能力
    """

    def __init__(self, in_channels: int, K: int, spatial_channels: int,
                 time_channels: int, edge_index: torch.Tensor,
                 time_strides: int, vertices_num: int, time_step_num: int,
                 device):
        """
        构造函数
        :param in_channels: 如通道数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param spatial_channels: 空间卷积后通道数量
        :param time_channels: 时间卷积后通道数量
        :param time_strides: 时间步
        :param vertices_num: 节点数量 N
        :param time_step_num: 时间步数量 T
        :param device: 模型所在设备
        """
        super(STACGINBlock, self).__init__()
        # 时间注意力层
        self.TAtt = TemporalAttentionLayer(
            in_channels=in_channels,
            vertices_num=vertices_num,
            time_steps_num=time_step_num
        )
        # 空间注意力层
        self.SAtt = SpatialAttentionLayer(
            in_channels=in_channels,
            vertices_num=vertices_num,
            time_steps_num=time_step_num
        )
        # GIN 空间卷积层
        self.SpatialConv = SpatialConvLayer(
            K=K,
            edge_index=edge_index,
            in_channels=in_channels,
            out_channels=spatial_channels,
            device=device
        )
        # 时间卷积层
        self.TimeConv = nn.Conv2d(
            in_channels=spatial_channels,
            out_channels=time_channels,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1)
        )
        # 残差卷积层
        self.ResidualConv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=time_channels,
            kernel_size=(1, 1),
            stride=(1, time_strides)
        )
        # 层归一化
        self.layer_norm = nn.LayerNorm(time_channels)

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, time_channels, T)
        """
        batch_size, vertices_num, features_num, time_step_num = x_matrix.size()

        # 计算时间注意力
        temporal_att = self.TAtt(x_matrix)  # (B, T, T)
        # 空间注意力处理原数据
        # 动态变化过程 (B, N, F, T) -reshape-> (B, N x F, T) -attention-> (B, N x F, T)
        # (B, N x F, T) -reshape-> (B, N, F, T)
        x_mat_t_att = torch.matmul(
            x_matrix.reshape(batch_size, -1, time_step_num),
            temporal_att
        ).reshape(batch_size, vertices_num, features_num, time_step_num)

        # 计算空间注意力，与时间注意力同理
        spatial_att = self.SAtt(x_mat_t_att)  # (B, N, N)
        x_mat_s_att = torch.matmul(
            spatial_att,
            x_mat_t_att.reshape(batch_size, vertices_num, -1)
        ).reshape(batch_size, vertices_num, features_num, time_step_num)

        # GIN 空间卷积
        x_mat_cheb_conv = self.SpatialConv(x_mat_s_att)  # (B, N, spatial_channels, T)

        # 时间卷积
        x_mat_time_conv = self.TimeConv(x_mat_cheb_conv.permute(0, 2, 1, 3))  # (B, N, time_channels, T)

        # 残差分支计算
        x_mat_resi = self.ResidualConv(x_matrix.permute(0, 2, 1, 3))  # (B, time_channels, N, T)

        # 残差连接
        # 动态变化过程 (B, TC, N, T) + (B, TC, N, T) -permute-> (B, T, N, TC)
        # (B, T, N, TC) -permute-> (B, N, TC, T)
        x_out = self.layer_norm(
            F.relu(x_mat_resi + x_mat_time_conv).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)

        return x_out


class STACGIN(nn.Module):
    """
    STACGIN whole module
    STACGIN 完整模型
    """

    def __init__(self, block_num: int, in_channels: int, K: int,
                 edge_index: torch.Tensor, spatial_channels: int,
                 time_channels: int, time_strides: int, predict_steps: int,
                 input_steps: int, vertices_num: int,
                 in_features: int, device):
        """
        构造函数
        :param block_num: STACGIN 块数量
        :param in_channels: 入通道数，这里是 Policy Conv后的通道数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param spatial_channels: chebyshev 卷积输出通道数
        :param time_channels: 时间卷积输出通道数
        :param time_strides: 时间卷积步长
        :param predict_steps: 预测时间步
        :param input_steps: 输入时间步
        :param vertices_num: 节点数量
        :param edge_index: 邻接矩阵对应的双列表表示
        :param in_features: 输入特征数，最外层的特征数
        :param device: 模型所在设备
        """
        super(STACGIN, self).__init__()

        # 政策卷积层
        self.PolicyConv = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_channels,
            kernel_size=(1, 1),
            stride=(1, time_strides)
        )
        # 默认模型
        self.BlockList = nn.ModuleList([
            STACGINBlock(
                in_channels=in_channels,
                K=K,
                time_channels=time_channels,
                edge_index=edge_index,
                time_strides=time_strides,
                vertices_num=vertices_num,
                time_step_num=input_steps,
                device=device,
                spatial_channels=spatial_channels
            )
        ])
        # 扩展模型
        self.BlockList.extend([
            STACGINBlock(
                in_channels=time_channels,
                K=K,
                spatial_channels=spatial_channels,
                time_channels=time_channels,
                edge_index=edge_index,
                time_strides=1,
                vertices_num=vertices_num,
                time_step_num=input_steps // time_strides,
                device=device
            ) for _ in range(block_num - 1)
        ])
        # 输出卷积层
        self.FinalConv = nn.Conv2d(
            in_channels=int(input_steps / time_strides),
            out_channels=predict_steps,
            kernel_size=(1, time_channels),
        )

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, T) 单个特征
        """

        # 计算 Policy Conv 块
        # 动态变化情况 (B, N, F_in, T) -permute-> (B, F_in, N, T)
        # (B, F_in, N, T) -conv-> (B, c_in, N, T) -permute-> (B, N ,c_in, T)
        x_matrix = self.PolicyConv(
            x_matrix.permute((0, 2, 1, 3))
        ).permute((0, 2, 1, 3))

        # 计算通过 ASTGCN 块
        for block in self.BlockList:
            x_matrix = block(x_matrix)  # (B, N, TC, T)

        # 计算最终预测输出
        # 动态变化情况 (B, N, TC, T) -permute-> (B, T, N, TC)
        # 卷掉了 Feature 维度特征
        # (B, T, N, TC) -Conv<1, tc>-> (B, T_c_out, N, 1) -squeeze-> (B, T_c_out, N)
        # (B, T_c_out, N) -permute-> (B, N, T_out)
        x_mat_output = self.FinalConv(
            x_matrix.permute(0, 3, 1, 2)
        ).squeeze(-1).permute(0, 2, 1)

        return x_mat_output


def make_model(
        block_num: int, in_channels: int, K: int, spatial_channels: int,
        time_channels: int, time_strides: int, predict_step: int, in_features: int,
        input_step: int, vertices_num: int, edge_index: torch.Tensor, device
):
    """
    模型的 Make 方法
    :param in_features: 输入特征数
    :param edge_index: 邻接矩阵边集
    :param device: 模型保存的设备
    :param block_num: ASTGCN 块数
    :param in_channels: 输入通道数
    :param K: 阶数
    :param spatial_channels: chebyshev卷积输出通道数
    :param time_channels: 时间卷积输出通道数
    :param time_strides: 时间步
    :param predict_step: 预测时间步
    :param input_step: 输入时间步
    :param vertices_num: 节点数量
    :return: 构造好的模型
    """

    model = STACGIN(
        block_num=block_num,
        in_channels=in_channels,
        K=K,
        spatial_channels=spatial_channels,
        time_channels=time_channels,
        time_strides=time_strides,
        predict_steps=predict_step,
        input_steps=input_step,
        vertices_num=vertices_num,
        device=device,
        edge_index=edge_index,
        in_features=in_features
    ).to(device)

    # 模型参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


if __name__ == '__main__':
    vertices_num = 307
    features_num = 10
    in_channels = 3
    time_step_num = 12
    chebyshev_channels = 64
    time_channels = 64
    batch_size = 10
    time_strides = 1
    predict_steps = 12
    input_steps = 12
    K = 3
    data_dir = '../data/PEMS04/PEMS04.csv'
    block_num = 2
    X = torch.rand((batch_size, vertices_num, features_num, time_step_num))
    print('input X shape: {}'.format(X.size()))

    edge_index = get_edge_index(data_dir)

    model = make_model(
        block_num=block_num,
        in_channels=in_channels,
        K=K,
        spatial_channels=chebyshev_channels,
        time_channels=time_channels,
        time_strides=time_strides,
        vertices_num=vertices_num,
        input_step=input_steps,
        predict_step=predict_steps,
        device=torch.device('cpu'),
        edge_index=torch.from_numpy(edge_index).type(torch.long),
        in_features=features_num
    )
    output_mat = model(X)

    print('output X shape: {}'.format(output_mat.size()))
    print('parameters: \n {}'.format(get_parameter_number(model)))
