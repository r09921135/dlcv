import torch
import torch.nn as nn
import torchvision
from torchvision import models


class Model(nn.Module):
    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        num_class = 65

        resnet = models.resnet50(pretrained=False)
        if pretrained != None:
            checkpoint = torch.load(pretrained)
            resnet.load_state_dict(checkpoint)
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Linear(128, num_class)
        )

    def forward(self, input):
        # input: (B, C, H, W)
        x = self.backbone(input)  # (B, C, 1, 1)
        x = x.reshape(-1, x.shape[1])  # (B, C)
        x = self.classifier(x)  # (B, 65)

        return x


if __name__ == '__main__':
    model = Model('best_model.pth.tar')
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print('model output shape:', y.shape)
