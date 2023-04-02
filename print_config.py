import platform
import torch


def print_platform_config():
    print('Hello Highway Flow Forecast ASTGCN')
    print('========Configuration========')
    print(f'Python Version: {platform.python_version()}')
    print(f'Pytorch CUDA Available: {torch.cuda.is_available()}')


if __name__ == '__main__':
    print_platform_config()
