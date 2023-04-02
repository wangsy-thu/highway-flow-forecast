from models.astgcn import make_model
from utils.chebyshev_conv import make_chebyshev_polynomial, get_adjacency_matrix, scaled_Laplacian
import torch
from tensorboardX import SummaryWriter

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
    data_dir = './data/PEMS04/PEMS04.csv'
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

    with SummaryWriter(logdir='./experiments/PEMS04/astgcn_r_h1d0w0_channel3_1.000000e-03',
                       flush_secs=5) as sw:
        sw.add_graph(model=model, input_to_model=X)
    print('=====Model Visualization Success=====')
