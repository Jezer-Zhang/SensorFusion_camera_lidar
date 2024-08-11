import random
import glob
import torch
import torch.utils.data as data


def split_dataset(dataset, training_percent, test_percent):
    dataset_length = len(dataset)  # 多少个tuple(ori,noi)
    train_size = int(training_percent * dataset_length)
    test_size = int(test_percent * dataset_length)
    val_size = dataset_length - train_size - test_size
    # print(len(dataset)) ---404 所有ori_img总数
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(0))
    return train_dataset, test_dataset, val_dataset


if __name__ == '__main__':
    pass
