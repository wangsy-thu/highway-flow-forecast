import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.chebyshev_conv import make_chebyshev_polynomial, scaled_Laplacian, get_adjacency_matrix
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


class ChebyshevAttentionConv(nn.Module):
    """
    Chebyshev Conv Layer with Spatial Attention
    带有空间注意力机制的 Chebyshev 卷积层
    """

    def __init__(self, K: int, chebyshev_polynomials: list,
                 in_channels: int, out_channels: int, device):
        """
        构造函数
        :param K: chebyshev 卷积核阶数，代表聚集 K 跳邻居的 Message
        :param chebyshev_polynomials: chebyshev 多项式矩阵列表, 长度为 K
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param device: 模型所在设备
        """
        super(ChebyshevAttentionConv, self).__init__()
        self.K = K
        self.chebyshev_polynomials = chebyshev_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        # 切比雪夫多项式卷积过程 可学习的参数 theta列表
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)]
        )

    def forward(self, x_matrix, spatial_attention):
        """
        前向传播函数
        :param x_matrix: 输入矩阵数据 (B, N, F_in, T)
        :param spatial_attention: 空间注意力矩阵 (B, N, N)
        :return: 输出矩阵数据 (B, N, F_out, T)
        """

        batch_size, vertices_num, in_channel, time_step_num = x_matrix.size()
        outputs = []

        # 对每一个时间步，整合空间注意力做空间卷积
        for time_step in range(time_step_num):

            # 提取该时间步下的图信号
            graph_sig = x_matrix[:, :, :, time_step]

            # 自定义的中间变量，不参与模型的
            output = torch.zeros(batch_size, vertices_num, self.out_channels).to(self.device)

            # 对 Graph 进行 K 跳邻居卷积
            for k in range(self.K):
                T_k = self.chebyshev_polynomials[k]  # (N, N)
                # 添加 Spatial Attention
                T_k_att = torch.matmul(T_k, spatial_attention)  # (N, N)
                # 整合可学习参数
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                tmp_res = torch.matmul(T_k_att.permute(0, 2, 1), graph_sig)  # (B, N, F_in)
                output = output + torch.matmul(tmp_res, theta_k)  # (B, N, F_out)

            # 堆叠输出
            outputs.append(output.unsqueeze(-1))  # (B, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCNBlock(nn.Module):
    """
    ASTGCN Block
    """

    def __init__(self, in_channels: int, K: int, chebyshev_channels: int,
                 time_channels: int, chebyshev_polynomials: list,
                 time_strides: int, vertices_num: int, time_step_num: int,
                 device):
        """
        构造函数
        :param in_channels: 如通道数
        :param K: chebyshev 卷积阶数
        :param chebyshev_channels: chebyshev卷积后的通道数
        :param time_channels: 时间卷积后通道数量
        :param chebyshev_polynomials: chebyshev卷积核列表，本质上就是矩阵列表
        :param time_strides: 时间步
        :param vertices_num: 节点数量 N
        :param time_step_num: 时间步数量 T
        :param device: 模型所在设备
        """
        super(ASTGCNBlock, self).__init__()
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
        # 切比雪夫卷积层
        self.ChebConv = ChebyshevAttentionConv(
            K=K,
            chebyshev_polynomials=chebyshev_polynomials,
            in_channels=in_channels,
            out_channels=chebyshev_channels,
            device=device
        )
        # 时间卷积层
        self.TimeConv = nn.Conv2d(
            in_channels=chebyshev_channels,
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

        # 计算空间注意力
        spatial_att = self.SAtt(x_mat_t_att)  # (B, N, N)

        # 切比雪夫卷积配合空间注意力
        x_mat_cheb_conv = self.ChebConv(x_matrix, spatial_att)  # (B, N, cheb_channels, T)

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


class ASTGCN(nn.Module):
    """
    ASTGCN whole module
    ASTGCN 完整模型
    """

    def __init__(self, block_num: int, in_channels: int, K: int,
                 chebyshev_channels: int, time_channels: int, time_strides: int,
                 chebyshev_polynomials: list, predict_steps: int,
                 input_steps: int, vertices_num: int, device):
        """
        构造函数
        :param block_num: ASTGCN 块数量
        :param in_channels: 入通道数，这里指特征数
        :param K: chebyshev 卷积核阶数
        :param chebyshev_channels: chebyshev 卷积输出通道数
        :param time_channels: 时间卷积输出通道数
        :param time_strides: 时间卷积步长
        :param chebyshev_polynomials: chebyshev 多项式，(N, N)矩阵序列
        :param predict_steps: 预测时间步
        :param input_steps: 输入时间步
        :param vertices_num: 节点数量
        :param device: 模型所在设备
        """
        super(ASTGCN, self).__init__()
        # 默认模型
        self.BlockList = nn.ModuleList([
            ASTGCNBlock(
                in_channels=in_channels,
                K=K,
                chebyshev_channels=chebyshev_channels,
                time_channels=time_channels,
                chebyshev_polynomials=chebyshev_polynomials,
                time_strides=time_strides,
                vertices_num=vertices_num,
                time_step_num=input_steps,
                device=device
            )
        ])
        # 扩展模型
        self.BlockList.extend([
            ASTGCNBlock(
                in_channels=time_channels,
                K=K,
                chebyshev_channels=chebyshev_channels,
                time_channels=time_channels,
                chebyshev_polynomials=chebyshev_polynomials,
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
        block_num: int, in_channels: int, K: int, chebyshev_channels: int,
        time_channels: int, time_strides: int, adj_mat, predict_step: int,
        input_step: int, vertices_num: int, device
):
    """
    模型的 Make 方法
    :param device: 模型保存的设备
    :param block_num: ASTGCN 块数
    :param in_channels: 输入通道数
    :param K: 阶数
    :param chebyshev_channels: chebyshev卷积输出通道数
    :param time_channels: 时间卷积输出通道数
    :param time_strides: 时间步
    :param adj_mat: 邻接矩阵
    :param predict_step: 预测时间步
    :param input_step: 输入时间步
    :param vertices_num: 节点数量
    :return: 构造好的模型
    """
    laplace_mat = scaled_Laplacian(adj_mat)
    chebyshev_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                             for i in make_chebyshev_polynomial(laplace_mat, K)]
    model = ASTGCN(
        block_num=block_num,
        in_channels=in_channels,
        K=K,
        chebyshev_channels=chebyshev_channels,
        time_channels=time_channels,
        time_strides=time_strides,
        chebyshev_polynomials=chebyshev_polynomials,
        predict_steps=predict_step,
        input_steps=input_step,
        vertices_num=vertices_num,
        device=device
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
    features_num = 3
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

    adj_mat, _ = get_adjacency_matrix(data_dir, vertices_num)
    laplace_mat = scaled_Laplacian(adj_mat)
    chebyshev_polynomials = [torch.from_numpy(i).type(torch.FloatTensor)
                             for i in make_chebyshev_polynomial(laplace_mat, K)]

    model = make_model(
        block_num=block_num,
        in_channels=features_num,
        K=K,
        chebyshev_channels=chebyshev_channels,
        time_channels=time_channels,
        time_strides=time_strides,
        vertices_num=vertices_num,
        adj_mat=adj_mat,
        input_step=input_steps,
        predict_step=predict_steps,
        device=torch.device('cpu')
    )
    output_mat = model(X)
    print('output X shape: {}'.format(output_mat.size()))
    print('parameters: {}\n'.format(get_parameter_number(model)))
