import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    npz_data = np.load('../../data/SZ/SZ.npz')
    all_time_data = npz_data['data']

    # 1-plot [one node, all feature, all time] figure
    single_node_data = all_time_data[:, 0, :]
    print(single_node_data.shape)
    plt.title('time-feature0')
    plt.plot([i for i in range(single_node_data.shape[0])], single_node_data[:, 0])
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.savefig('./plot_feature/time-feature0.png')
    plt.figure()

    # 2-plot [all node, all feature, single time] figure
    single_time_data = all_time_data[0, :, :]
    print(single_time_data.shape)
    plt.title('node-feature0')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 0])
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.savefig('./plot_feature/node-feature0.png')
    plt.figure()
