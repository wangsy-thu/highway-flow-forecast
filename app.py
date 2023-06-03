import argparse
import configparser
import os

import torch
from flask import Flask, make_response
from flask import request
import numpy as np

from models.stacgin import make_model
from utils.adj import get_edge_index
from utils.web_utils import make_common_response

app = Flask(__name__)
print('===== Loading Result Data =====')
# 1, 加载预测结果的 Numpy 数据到内存
if os.path.exists('./workspace/forecast-result.npz'):
    res_mat = np.load('./workspace/forecast-result.npz')['data']
# 2, 加载历史数据的 Numpy 数据到内存
if os.path.exists('./workspace/history-flow.npz'):
    history_mat = np.load('./workspace/history-flow.npz')['data']
# 3, 这里以 PEMS04 数据集为例拼接
res = np.concatenate((history_mat[:, :, 0], res_mat), axis=0)
print('===== Result Data Load Success =====')

# 4, 加载预测模型
print('===== Loading Forecast Model =====')
# 1,解析参数与配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config/PEMS04_stacgin.conf', type=str,
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
# file_data = np.load('./data/LOS/LOS_r1_d0_w0_stacgin.npz')
# mean, std = file_data['mean'], file_data['std']

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

# params_file_name = os.path.join(params_path, 'epoch_%s.params' % '78')
# stacgin_net.load_state_dict(torch.load(params_file_name, map_location='cpu'))
stacgin_net.train(False)
print('===== Model Load Success =====')


@app.after_request
def func_res(resp):
    """
    跨域处理统一接口
    :param resp: 响应数据
    :return: 经过跨域处理后的响应
    """
    res = make_response(resp)
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTION'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return res


@app.route('/query-forecast-result', methods=['GET'])
def query_forecast_result():
    """
    查询预测结果
    :return: Common Resp
    """
    node_id = request.args.get('nodeId')
    flow_list: np.ndarray = res[:, int(node_id)]
    return make_common_response(
        state=0,
        message='ok',
        data=flow_list.tolist()
    )


@app.route('/forecast-npz-flow', methods=['GET'])
def forecast_npz_flow():
    """
    预测流量
    :return: Common Resp
    """
    # # 1, 读取待预测文件
    # history_mat = np.load('./workspace/history-flow.npz')['data']
    #
    # # 2, 生成预测结果
    # input_flow = torch.from_numpy(history_mat).float().unsqueeze(0)
    # input_norm = (input_flow - torch.from_numpy(mean)) / torch.from_numpy(std)
    # with torch.no_grad():
    #     result: torch.Tensor = stacgin_net(input_norm)
    #
    # # 3, 将预测结果写回文件中
    # res_mat = result.permute(0, 2, 1).numpy()[0, :, :]  # 去掉 batch_size 维度
    # np.savez_compressed('./workspace/forecast-result.npz',
    #                     data = res_mat)

    # 4, 返回预测结果
    return make_common_response(
        state=0,
        message='ok',
        data={}
    )


@app.route('/upload-history-flow', methods=['POST'])
def upload_history_flow():
    """
    上传历史数据文件
    :return: Common Resp
    """
    request.files.get('file').save('./workspace/history-flow.npz')
    return make_common_response(
        state=0,
        message='ok',
        data={}
    )


@app.route('/clear-cache', methods=['GET'])
def clear_cache():
    """
    清空缓存区，删除上传的流量文件与结果文件
    :return: Common Resp
    """
    file_path = './workspace/'
    for file in os.listdir(file_path):
        os.remove(file_path + file)

    return make_common_response(
        state=0,
        message='ok',
        data={}
    )



if __name__ == '__main__':
    # Flask 启动入口
    app.run()
