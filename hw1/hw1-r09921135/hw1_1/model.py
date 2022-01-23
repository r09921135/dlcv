import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, pretrained=False):
        super(Model, self).__init__()
        num_class = 50

        # resnet
        model = torchvision.models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(model.children())[:9])
        self.classifier = nn.Linear(512, num_class)

        # vgg
        # model = torchvision.models.vgg16(pretrained=False)
        # self.backbone = model.features
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, num_class),
        # )

    def forward(self, input):
        # input: (B, C, H, W)
        x = self.backbone(input)  # (B, 512, 1, 1)
        x = x.reshape(-1, x.shape[1])  # (B, 512)
        x = self.classifier(x)  # (B, 50)
        return x


if __name__ == '__main__':
    model = Model()
    # print(model)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print('model output shape:', y.shape)