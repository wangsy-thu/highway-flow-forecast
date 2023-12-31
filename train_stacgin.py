import argparse
import configparser
import os
import shutil

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.make_data_loader import make_flow_data_loader
from models.stacgin import make_model
from utils.adj import get_edge_index
from utils.inference import predict_and_save_results
from utils.loss import compute_val_loss
from utils.metrics import masked_mae, masked_mse

# 1,解析参数与配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config/LN_stacgin.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("Cuda Available:{}, use {}!".format(USE_CUDA, DEVICE))

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
spatial_channels = int(training_config['nb_spatial_filter'])
time_channels = int(training_config['nb_temporal_filter'])
in_channels = int(training_config['in_channels'])
in_features = int(training_config['in_features'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' \
             % (model_name, num_of_hours, num_of_days,
                num_of_weeks, in_channels, learning_rate)
params_path = os.path.join('./experiments', dataset_name, folder_dir)

edge_index = get_edge_index(adj_filename)
print('batch size: {}'.format(batch_size))

# 2,初始化模型与数据集
train_loader, val_loader, test_loader, test_target_tensor, _mean, _std = make_flow_data_loader(
    flow_matrix_filename=graph_signal_matrix_filename,
    num_of_weeks=num_of_weeks,
    num_of_days=num_of_days,
    num_of_hours=num_of_hours,
    batch_size=batch_size,
    device=DEVICE,
    model_name=model_name,
    shuffle=True
)

stacgin_net = make_model(
    block_num=nb_block,
    in_channels=in_channels,
    K=K,
    spatial_channels=spatial_channels,
    time_channels=time_channels,
    time_strides=time_strides,
    in_features=in_features,
    predict_step=num_for_predict,
    input_step=len_input,
    vertices_num=num_of_vertices,
    device=DEVICE,
    edge_index=torch.from_numpy(edge_index).type(torch.long).to(DEVICE)
)


def train_main():
    """
    Train the Model
    """

    # 1, 解析训练参数
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        # 从头训练且不存在参数文件夹 -> 创建参数文件夹
        os.makedirs(params_path)
        print('Create params directory {}'.format(params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        # 从头训练且存在参数文件夹 -> 先删除，后创建
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Delete Old directory, Create params directory {}'.format(params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        # 断点续训 -> 后续选择
        print('Train from params directory')
    else:
        # 其余任何情况均为非法
        raise SystemExit('Wrong Hyper Params')

    # 2,定义训练要素: Loss, Optimizer, Model
    criterion = nn.MSELoss().to(DEVICE)  # MSE 损失函数
    optimizer = optim.Adam(stacgin_net.parameters(), lr=learning_rate)  # Adam优化器
    criterion_masked = masked_mae
    masked_flag = 0
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    sw = SummaryWriter(logdir=params_path, flush_secs=5)  # 训练日志监控工具

    # 3, 加载断点续训模型
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:
        params_file_name = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        stacgin_net.load_state_dict(torch.load(params_file_name, map_location=DEVICE))
        print('Start Epoch: {}'.format(start_epoch))

    # 4,训练模型
    print('=====Start Training=====')
    for epoch in range(start_epoch, epochs):

        # 1,判断是否保存上次训练后的模型
        params_file_name = os.path.join(params_path, 'epoch_%s.params' % epoch)
        if masked_flag:
            val_loss = compute_val_loss(
                net=stacgin_net,
                val_loader=val_loader,
                criterion=criterion_masked,
                masked_flag=masked_flag,
                missing_value=missing_value,
                sw=sw,
                epoch=epoch
            )
        else:
            val_loss = compute_val_loss(
                net=stacgin_net,
                val_loader=val_loader,
                criterion=criterion,
                masked_flag=masked_flag,
                missing_value=missing_value,
                sw=sw,
                epoch=epoch
            )

        # 找到最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(stacgin_net.state_dict(), params_file_name)
            print('Save Model to: {}'.format(params_path))

        stacgin_net.train()  # 恢复训练模式，开启梯度更新
        print('=====Epoch [{}]====='.format(epoch + 1))
        for batch_data in tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1)):
            encoder_input, labels = batch_data

            # 训练模型
            optimizer.zero_grad()  # 梯度清零
            outputs = stacgin_net(encoder_input)  # 正向传播
            # 计算损失
            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()  # 反向求导
            optimizer.step()  # 梯度更新

            train_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', train_loss, global_step)

    print('=====Train Finished=====')
    print('=====Best Epoch: {}====='.format(best_epoch))
    print('=====Predict Start=====')
    predict_main(
        global_step=best_epoch,
        data_loader=test_loader,
        test_target_tensor=test_target_tensor,
        metric_method=metric_method,
        _mean=_mean,
        _std=_std,
        type='test',
        sw=sw
    )
    print('=====Predict Success=====')


def predict_main(global_step: int, data_loader: DataLoader,
                 test_target_tensor: torch.Tensor,
                 metric_method: str, _mean, _std, type: str,
                 sw: SummaryWriter):
    """
    Predict 预测模型搭建
    :param sw: 日志工具
    :param global_step: 使用哪步作为参数
    :param data_loader: 数据加载器
    :param test_target_tensor: 标签矩阵
    :param metric_method: 评价标准方法
    :param _mean: 均值
    :param _std: 方差
    :param type: 类型
    """

    params_file_name = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('Loading Model From: {}'.format(params_file_name))
    stacgin_net.load_state_dict(torch.load(params_file_name, map_location='cpu'))
    predict_and_save_results(
        net=stacgin_net,
        data_loader=data_loader,
        data_target_tensor=test_target_tensor,
        global_step=global_step,
        metric_method=metric_method,
        _mean=_mean,
        _std=_std,
        params_path=params_path,
        type=type,
        sw=sw,
        plot_sensor_count=10
    )


if __name__ == '__main__':
    # train model
    train_main()
