import torch
import torch.nn as nn
import torchvision

class FCN32(nn.Module):
    def __init__(self, pretrained=False):
        super(FCN32, self).__init__()
        self.num_class = 7
        model = torchvision.models.vgg16(pretrained=pretrained)
        self.backbone = model.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, self.num_class, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(self.num_class, self.num_class, 64 , 32),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.backbone(x)  # (B, 512, 8, 8)
        # print(x.shape)
        x = self.classifier(x)  # (B, 50, H, W)
        return x

    
class FCN16(nn.Module):
    def __init__(self, pretrained=False):
        super(FCN16, self).__init__()
        self.num_class = 7
        model = torchvision.models.vgg16(pretrained=pretrained)
        self.to_pool4 = nn.Sequential(*list(model.features[:24]))
        self.to_pool5 = nn.Sequential(*list(model.features[24:]))
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, self.num_class, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(self.num_class, 512, 4, 2)
            )
        self.upsample16 = nn.ConvTranspose2d(512, self.num_class, 16, 16)
                                      
    def forward (self, x) :      
        # x: (B, C, H, W)
        pool4_output = self.to_pool4(x)  # (B, 512, 16, 16)
        x = self.to_pool5(pool4_output)  # (B, 512, 8, 8)
        x = self.classifier(x)  # (B, 512, 16, 16)
        x = self.upsample16(x + pool4_output)  # (B, 50, H, W)
        return x


class FCN8(nn.Module):
    def __init__(self, pretrained=False):
        super(FCN8, self).__init__()
        self.num_class = 7
        model = torchvision.models.vgg16(pretrained=pretrained)
        self.to_pool3 = nn.Sequential(*list(model.features[:17]))
        self.to_pool4 = nn.Sequential(*list(model.features[17:24]))
        self.to_pool5 = nn.Sequential(*list(model.features[24:]))
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, self.num_class, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(self.num_class, 256, 8 , 4)
            )
        self.pool4_upsample2 = nn.ConvTranspose2d(512, 256, 2 ,2)
        self.upsample8 = nn.ConvTranspose2d(256, self.num_class, 8 , 8)
        
    def forward (self, x) :      
        # x: (B, C, H, W)
        pool3_output = self.to_pool3(x)  # (B), 256, 32, 32)
        pool4_output = self.to_pool4(pool3_output)  # (B, 512, 16, 16)
        pool4_2x = self.pool4_upsample2(pool4_output)  # (B, 512, 32, 32)
        x = self.to_pool5(pool4_output)  # (B, 512, 8, 8)
        x = self.classifier(x)  # (B, 256, 32, 32)
        x = self.upsample8(x + pool3_output + pool4_2x)  # (B, 7, H, W)
        return x


if __name__ == '__main__':
    model = FCN8()
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print('output shape from model:', y.shape)
    # print(model)