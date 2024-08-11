from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transform
from PIL import Image
import glob
import cv2
import numpy as np


# customize dataset, return ori and noi image to a tuple
class CreateDatasets(Dataset):
    def __init__(self, label_list, damaged_img_list, lidar_list):
        self.label_list = label_list
        self.damaged_img_list = damaged_img_list
        self.lidar_list = lidar_list
        self.transform = transform.Compose([
            transform.ToTensor()
            # transform.Resize((img_size, img_size)),
            # transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.label_list)
    # 此处只能return一个！

    def __getitem__(self, idx):
        label = Image.open(self.label_list[idx])
        damaged_img = Image.open(self.damaged_img_list[idx])
        lidar = torch.load(self.lidar_list[idx])
        label = self.transform(label.copy())
        damaged_img = self.transform(damaged_img)
        return label, damaged_img, lidar


if __name__ == '__main__':
    damage_label = 'A'
    ori_img_path = 'D:/Sensor_fusion_dataset/Dataset'
    noi_img_path = f'D:/Sensor_fusion_dataset/Damaged_dataset/damage_{damage_label}'
    ori_img_list = sorted(glob.glob(ori_img_path + '/*'))
    noi_img_list = sorted(glob.glob(noi_img_path + '/*'))
    datasetA = CreateDatasets(ori_img_list, noi_img_list, img_size=256)

    # print(len(datasetA.__getitem__(0)))
    # print(datasetA.ori_img_list[0], noi_img_list[0])
    # print(datasetA.__len__()[0])

    # img =datasetA.__getitem__(0)[0]
    # img1 = datasetA.__getitem__(0)[1]
    # img.show()
    # img1.show()

    # customize dataset, return ori and noi image to a tuple
