from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transform
from PIL import Image
import glob
import cv2
import numpy as np


# customize dataset, return ori and noi image to a tuple
class CreateDatasets(Dataset):
    def __init__(self, ori_img_list, noi_img_list):
        self.ori_img_list = ori_img_list
        self.noi_img_list = noi_img_list
        self.transform = transform.Compose([
            transform.ToTensor()
            # transform.Resize((img_size, img_size)),
            # transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_img_list)
    # 此处只能return一个！

    def __getitem__(self, idx):
        ori_img = Image.open(self.ori_img_list[idx])
        noi_img = torch.load(self.noi_img_list[idx])
        ori_img = self.transform(ori_img.copy())
        return ori_img, noi_img


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
