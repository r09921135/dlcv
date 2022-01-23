import torch
import torch.nn as nn
import torchvision
from pytorch_pretrained_vit import ViT


class Model(nn.Module):
    def __init__(self, pretrained=False):
        super(Model, self).__init__()
        num_class = 37

        self.backbone = ViT('B_16_imagenet1k', pretrained=pretrained)
        self.backbone.fc = nn.Linear(768, num_class)

    def forward(self, input):
        # input: (B, C, H, W)
        x = self.backbone(input)

        return x


if __name__ == '__main__':
    model = Model()
    breakpoint()
    x = torch.rand(1, 3, 384, 384)
    y = model(x)
    print('model output shape:', y.shape)
