from models.stacgin import make_model
import torch
from tensorboardX import SummaryWriter
from utils.adj import get_edge_index

if __name__ == '__main__':
    vertices_num = 307
    features_num = 3
    in_channels = 3
    time_step_num = 12
    spatial_channels = 64
    time_channels = 64
    batch_size = 10
    time_strides = 1
    predict_steps = 12
    input_steps = 12
    K = 3
    data_dir = './data/PEMS04/PEMS04.csv'
    block_num = 2
    X = torch.rand((batch_size, vertices_num, features_num, time_step_num))
    print('input X shape: {}'.format(X.size()))

    edge_index = get_edge_index(data_dir)

    model = make_model(
        block_num=2,
        in_channels=in_channels,
        K=K,
        spatial_channels=spatial_channels,
        time_channels=time_channels,
        time_strides=time_strides,
        in_features=features_num,
        predict_step=predict_steps,
        input_step=input_steps,
        vertices_num=vertices_num,
        device=torch.device('cpu'),
        edge_index=torch.from_numpy(edge_index).type(torch.long)
    )
    output_mat = model(X)

    with SummaryWriter(logdir='./experiments/PEMS04/stacgin_r_h1d0w0_channel3_1.000000e-03',
                       flush_secs=5) as sw:
        sw.add_graph(model=model, input_to_model=X)
    print('=====Model Visualization Success=====')
