# Highway Flow Forecast
## 一、STAC-GIN
Spatial Temporal Attention and Convolution based Graph Isomorphism Network(STAC-GIN) Highway Flow Forecast

基于时空注意力与时空卷积的图同构网络高速车流量预测算法

算法架构图如下:

<img src="./assets/structure.png" alt="">

Spatial Convolution Layer结构如下

<img src="./assets/spatial_conv.png" alt="">

## 二、Prepare Environment
```shell
# pytorch-gpu
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# pytorch-geometric
pip install torch_geometric

# tensorboardX
pip install tensorboardX

# networkx numpy pandas matplotlib scipy scikit-learn
pip install numpy pandas matplotlib networkx scipy scikit-learn

# tqdm
pip install tqdm
```

## 三、Train Model
```shell
# prepare data
python prepare_data.py --config=config/PEMS04_stacgin.conf

# train model
python train_stacgin.py --config=config/PEMS04_stacgin.conf
```

## 四、Trace Training
use TensorboardX to trace the training process
```shell
tensorboard --logdir={your log directory(full path)}
```