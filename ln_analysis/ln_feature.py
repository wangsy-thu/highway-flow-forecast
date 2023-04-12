import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    npz_data = np.load('../data/LN/LN.npz')
    all_time_data = npz_data['data']

    # 1-plot [one node, all feature, all time] figure
    single_node_data = all_time_data[:2880, 345, :]
    print(single_node_data.shape)
    plt.title('time-feature0')
    plt.plot([i for i in range(single_node_data.shape[0])], single_node_data[:, 0])
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.savefig('./plot_feature/time-feature0.png')
    plt.figure()

    plt.title('time-feature1')
    plt.plot([i for i in range(single_node_data.shape[0])], single_node_data[:, 1])
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.savefig('./plot_feature/time-feature1.png')
    plt.figure()

    plt.title('time-feature2')
    plt.plot([i for i in range(single_node_data.shape[0])], single_node_data[:, 2])
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.savefig('./plot_feature/time-feature2.png')
    plt.figure()

    plt.title('time-feature3')
    plt.plot([i for i in range(single_node_data.shape[0])], single_node_data[:, 3])
    plt.xlabel('time step')
    plt.ylabel('value')
    plt.savefig('./plot_feature/time-feature3.png')
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

    plt.title('node-feature1')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 1])
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.savefig('./plot_feature/node-feature1.png')
    plt.figure()

    plt.title('node-feature2')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 2])
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.savefig('./plot_feature/node-feature2.png')
    plt.figure()

    plt.title('node-feature3')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 3])
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.savefig('./plot_feature/node-feature3.png')
    plt.figure()

    plt.title('node-features')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 0], label='feature0')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 1], label='feature1')
    plt.plot([i for i in range(single_time_data.shape[0])], single_time_data[:, 2], label='feature2')
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('./plot_feature/node-features.png')
    plt.figure()

    print('=====Feature Map Done=====')
