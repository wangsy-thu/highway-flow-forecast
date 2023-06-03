import numpy as np


if __name__ == '__main__':
    npz_data = np.load('./data/PEMS04/PEMS04.npz')
    all_time_data = npz_data['data']

    offset = 1 * 12 * 24
    sample_history_data = all_time_data[offset: offset + 12, :, :]
    sample_forecast_data = all_time_data[offset + 12: offset + 24, :, 0]

    print('history_shape:{}'.format(sample_history_data.shape))
    print('forecast_shape:{}'.format(sample_forecast_data.shape))

    # save to npz
    np.savez_compressed(
        './workspace/history-flow.npz',
        data = sample_history_data
    )
    np.savez_compressed(
        './workspace/forecast-result.npz',
        data=sample_forecast_data
    )

    # validate result
    print('history_shape:{}'.format(np.load('./workspace/history-flow.npz')['data'].shape))
    print('forecast_shape:{}'.format(np.load('./workspace/forecast-result.npz')['data'].shape))
