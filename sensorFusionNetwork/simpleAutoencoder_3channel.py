import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


class ChannelChange(nn.Module):
    def __init__(self):
        super(channel_change, self).__init__()
        self.layer1 = nn.Conv2d(
                in_channels=128,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        
    def forward(self, x):
        x = self.layer1(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.downconv1 = nn.Sequential(
            # 输入[3,256,256]
            nn.Conv2d(
                in_channels=3,  # 输入图片的channel
                out_channels=32,  # 输出图片的channel,取决于kernel filter的数量
                kernel_size=5,  # 5x5的卷积核，相当于过滤器
                stride=1,  # 卷积核在图上滑动，每隔一个扫一次
                padding=2,  # 给图外边补上0
            ),
            # 经过卷积层 输出[32,256,256] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,128,128] 传入下一个卷积
        )

        self.downconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 同上
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[64, 128, 128] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[64,64,64] 传入输出层
        )

        self.encoderBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.encoderBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        x = self.downconv1(x)
        x = self.downconv2(x)
        x = x+self.encoderBlock1(x)
        x = x+self.encoderBlock2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.Sequential(  # 输入[64,64,64]
            nn.ConvTranspose2d(
                in_channels=64,  # 同上
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU()
        )
        # 输出[64, 128, 128]
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,  # 同上
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU()
        )
        # 输出[3, 256, 256]

        self.decoderBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # 同上
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=3,  # 同上
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.decoderBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # 同上
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=3,  # 同上
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = x+self.decoderBlock1(x)
        x = x+self.decoderBlock2(x)
        return x


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        # z = z + x 需要删除！不然最后增加noise!
        return z


if __name__ == '__main__':
    model = SimpleAutoencoder()
    # print(model)
    input = torch.ones(30, 3, 256, 256)
    output = model(input)
    print(output.shape)



# latent_dims = 2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# autoencoder = Autoencoder(latent_dims).to(device) # GPU


# class SimpleAutoencoder(nn.Module):
#     def __init__(self):
#         super(SimpleAutoencoder, self).__init__()
#         self.conv1 = nn.Sequential(
#             # 输入[3,256,256]
#             nn.Conv2d(
#                 in_channels=3,  # 输入图片的channel
#                 out_channels=16,  # 输出图片的channel,取决于kernel filter的数量
#                 kernel_size=5,  # 5x5的卷积核，相当于过滤器
#                 stride=1,  # 卷积核在图上滑动，每隔一个扫一次
#                 padding=2,  # 给图外边补上0
#             ),
#             # 经过卷积层 输出[16,256,256] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[16,128,128] 传入下一个卷积
#         )
#         # 第二层卷积
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,  # 同上
#                 out_channels=32,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2
#             ),
#             # 经过卷积 输出[32, 128, 128] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,64,64] 传入输出层
#         )
#
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)  # [batch, 32,7,7]
#         x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
#         output = self.output(x)  # 输出[50,10]
#         return output
