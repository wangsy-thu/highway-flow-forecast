import argparse
import configparser
import os

import numpy as np
import torch
import pika

from models.stacgin import make_model
from utils.adj import get_edge_index
from utils.protocol import byte2array, array2byte

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

# 2,加载训练时的均值与方差
file_data = np.load('./data/LOS/LOS_r1_d0_w0_stacgin.npz')
mean, std = file_data['mean'], file_data['std']

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

params_file_name = os.path.join(params_path, 'epoch_%s.params' % '78')
stacgin_net.load_state_dict(torch.load(params_file_name, map_location='cpu'))
stacgin_net.train(False)

# 3, 初始化 rabbitMQ 环境
user_auth = pika.PlainCredentials(
    username='WangY',
    password='WangY@20010418@WangY'
)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='101.42.231.27',
        credentials=user_auth
    )
)

channel = connection.channel()
channel.queue_declare('flow-data-channel')
channel.queue_declare('flow-predict-channel')
input_bytes = b''

def callback(ch, method, properties, body):
    print('=====Load Input=====')

    # 4, 从 MQ 中获取 input_flow, 并将 input_flow 处理成 torch.Tensor 类型
    input_flow = torch.from_numpy(byte2array(body)).float()
    input_norm = (input_flow - torch.from_numpy(mean)) / torch.from_numpy(std)

    with torch.no_grad():
        result: torch.Tensor = stacgin_net(input_norm)

    print(result.size())
    channel.basic_publish(
        exchange='',
        routing_key='flow-predict-channel',
        body=array2byte(result.numpy())
    )

# 开始监听消息队列
print('=====Predict LOS Service Start=====')
channel.basic_consume('flow-data-channel', callback, auto_ack=True)
channel.start_consuming()
