import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.downconv1 = nn.Sequential(
            # 输入[1,256,256] #[3,128,128]
            nn.Conv2d(
                in_channels=1,  # 输入图片的channel
                out_channels=32,  # 输出图片的channel,取决于kernel filter的数量
                kernel_size=5,  # 5x5的卷积核，相当于过滤器
                stride=1,  # 卷积核在图上滑动，每隔一个扫一次
                padding=2,  # 给图外边补上0
            ),
            # 经过卷积层 输出[32,256,256]  [32,128,128] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,128,128] [32,64,64] 传入下一个卷积
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
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[64,50,50] [64,32,32] 传入输出层
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
        self.upconv1 = nn.Sequential(  # 输入[64,50,50]
            nn.ConvTranspose2d(
                in_channels=64,  # 同上
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU()
        )
        # 输出[64, 128, 128]
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # 同上
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.ReLU()
        )
        # 输出[3, 256, 256]

        self.decoderBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 同上
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,  # 同上
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,  # 同上
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU()
        )

        self.decoderBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 同上
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,  # 同上
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = x+self.decoderBlock1(x)
        x = self.upconv3(x)
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

class EncoderFusion (nn.Module):
    def __init__(self):
        super(EncoderFusion, self).__init__()    
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # 同上
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,  # 同上
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        x = self.fusion(x)
        return x


    
class TwoEncoderOneDecoder(nn.Module):
    def __init__(self, encoder1, encoder2, decoder, channeladjust):
        super(TwoEncoderOneDecoder, self).__init__()
        self.encoder1 = encoder1 # already to device
        self.encoder2 = encoder2
        self.decoder = decoder
        self.channeladjust = channeladjust # not to device
        
    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z = torch.cat((z1,z2),1)
        z = self.channeladjust(z)
        z = self.decoder(z)
        
        return z
        
        
class WeightsGenerator_2(nn.Module):
    def __init__(self):
        super(WeightsGenerator_2, self).__init__()

        # Define the linear layers
        self.layer1 = nn.Sequential(
            nn.Linear(64 * 50 * 50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 50 * 50),
        )
            
        self.layer2 = nn.Sequential(
            nn.Linear(64 * 50 * 50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 50 * 50),
        )
            
    def forward(self, x1, x2):
        # print(x1.shape,x2.shape) --> torch.Size([32, 64, 50, 50]) torch.Size([32, 64, 50, 50])

        # Flatten the input tensors
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        # print(x1_flat.shape,x2_flat.shape) --> torch.Size([32, 160000]) torch.Size([32, 160000])

        # Pass the flattened input tensors through the linear layers
        w1 = self.layer1(x1_flat)
        w2 = self.layer2(x2_flat)
        # print(w1.shape,w2.shape)
        # Reshape the weight tensors back to the original shape
        w1 = w1.view(w1.size(0), 64, 50, 50)
        w2 = w2.view(w2.size(0), 64, 50, 50)
        # print(w1.shape,w2.shape) --> torch.Size([32, 64, 50, 50]) torch.Size([32, 64, 50, 50])

        # Multiply the output tensors with the input tensors element-wise and add them
        x3 = x1 * w1 + x2 * w2
        
        return x3




class WeightsGenerator(nn.Module):
    def __init__(self):
        super(WeightsGenerator, self).__init__()
        
#         # Define the convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        ) 
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),

#         )

#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         ) 
        # Add more convolutional layers as needed
        
        # # Define the output layers
        # self.output1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        # self.output2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
    def forward(self, x1, x2):
        # Apply the convolutional layers to the input tensors
        w1 = self.layer1(x1)
        # Add more convolutional layers as needed
        
        w2 = self.layer2(x2)
        # Add more convolutional layers as needed
        
        # Apply the output layers to get the weight tensors


        # Multiply the weight tensors with the input tensors element-wise and add them
        x3 = w1 * x1 + w2 * x2
        return x3

    
class TwoEncoderOneDecoder_fusion_4(nn.Module):
    def __init__(self, encoder1, encoder2, decoder, weights):
        super(TwoEncoderOneDecoder_fusion_4, self).__init__()
        self.encoder1 = encoder1 # already to device
        self.encoder2 = encoder2
        self.decoder = decoder
        self.weights = weights # not to device
        
    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z = self.weights(z1,z2)
        z = self.decoder(z)
        
        return z



class TwoEncoderOneDecoderAfterTraning(nn.Module):
    def __init__(self):
        super(TwoEncoderOneDecoderAfterTraning, self).__init__()
        self.encoder1 = Encoder() # already to device
        self.encoder2 = Encoder()
        self.decoder = Decoder()
        self.channeladjust = EncoderFusion() 
        
    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        z = torch.cat((z1,z2),1)
        z = self.channeladjust(z)
        z = self.decoder(z)

        return z 
    


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.k1 = nn.Parameter(torch.tensor(1.0))
        self.k2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, std_img, std_lidar, mean_img, mean_lidar):
        delta = self.P(std_img, std_lidar)
        decoder_input = self.fusion(delta, mean_img, mean_lidar)
        return decoder_input, self.k1, self.k2

    def P(self, std_img, std_lidar):
        delta = torch.log(self.k1 * (std_lidar / std_img)) * self.k2
        return delta

    def fusion(self, delta, mean_img, mean_lidar):
        w1 = torch.tanh(delta)
        w2 = 1 - w1
        decoder_input = w1 * mean_img + w2 * mean_lidar
        return decoder_input

class TwoEncoderOneDecoder_fusion_5(nn.Module):
    def __init__(self, encoder1, encoder2, decoder):
        super(TwoEncoderOneDecoder_fusion_5, self).__init__()
        self.regression = RegressionModel()
        self.encoder1 = encoder1 # already to device
        self.encoder2 = encoder2
        self.decoder = decoder

    def forward(self, img, lidar):
        hidden_img = self.encoder1(img)
        hidden_lidar = self.encoder2(lidar)
        hidden_img_mean = torch.mean(hidden_img, dim=0, keepdim=True)
        hidden_lidar_mean = torch.mean(hidden_lidar, dim=0, keepdim=True)
        hidden_img_std = torch.std(hidden_img, dim=0, keepdim=True)
        hidden_lidar_std = torch.std(hidden_lidar, dim=0, keepdim=True)
        delta, k1, k2 = self.regression(hidden_img_std, hidden_lidar_std, hidden_img_mean, hidden_lidar_mean)
        decoder_input = self.regression.fusion(delta, hidden_img_mean, hidden_lidar_mean)
        output = self.decoder(decoder_input)
        return output, k1, k2


        
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
