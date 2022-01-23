import torch
import torch.nn as nn
import torchvision.models as models
from functions import ReverseLayerF


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output 


class DANN_IMPROVED(nn.Module):
    """ Implementation of the DANN model """
    def __init__(self):
        super(DANN_IMPROVED, self).__init__()

        # The model feature extractor use resnet34 as backend
        self.resnet34 = models.resnet34(pretrained = False)
        self.resnet34 = nn.Sequential(*(list(self.resnet34.children())[:-2]))  # (B, 512, 1, 1)
        # for params in self.resnet34.parameters():   
        #     params.requires_grad = False
        self.resnet34.add_module('r_convt1', nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256, 
                                         kernel_size=4, 
                                         stride=2, padding=1, bias=False))  # (B, 256, 2, 2)
        self.resnet34.add_module('r_relu1', nn.ReLU(inplace=True))
        self.resnet34.add_module('r_convt2', nn.ConvTranspose2d(in_channels=256,
                                 out_channels=128, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False))   # (B, 128, 4, 4)
        self.resnet34.add_module('r_relu2', nn.ReLU(inplace=True))
        self.resnet34.add_module('r_convt3', nn.ConvTranspose2d(in_channels=128,
                                 out_channels=64, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False))  # (B, 64, 8, 8)
        self.resnet34.add_module('r_relu3', nn.ReLU(inplace=True))
        self.resnet34.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))   # (B, 50, 4, 4)
        self.resnet34.add_module('f_bn2', nn.BatchNorm2d(50))
        self.resnet34.add_module('f_drop1', nn.Dropout2d())
        self.resnet34.add_module('f_relu2', nn.ReLU(True))


        # This part of the cnn acts as an class classifier
        self.classify_class = nn.Sequential()
        self.classify_class.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.classify_class.add_module('c_bn1', nn.BatchNorm1d(100))
        self.classify_class.add_module('c_relu1', nn.ReLU(True))
        self.classify_class.add_module('c_drop1', nn.Dropout2d())
        self.classify_class.add_module('c_fc2', nn.Linear(100, 100))
        self.classify_class.add_module('c_bn2', nn.BatchNorm1d(100))
        self.classify_class.add_module('c_relu2', nn.ReLU(True))
        self.classify_class.add_module('c_fc3', nn.Linear(100, 10))
        self.classify_class.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # This part of the CNN acts as a domain classifier
        self.classify_domain = nn.Sequential()
        self.classify_domain.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.classify_domain.add_module('d_bn1', nn.BatchNorm1d(100))
        self.classify_domain.add_module('d_relu1', nn.ReLU(True))
        self.classify_domain.add_module('d_fc2', nn.Linear(100, 2))
        self.classify_domain.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        output = self.resnet34(input_data)
        output = output.view(-1, 50 * 4 * 4)

        reverse_feature = ReverseLayerF.apply(output, alpha)
        out_class = self.classify_class(output)
        out_domain = self.classify_domain(reverse_feature)

        return out_class, out_domain


if __name__ == '__main__':
    x = torch.rand(5, 3, 28, 28)
    m = DANN_IMPROVED()
    y = m(x, 1)
    print(y[0].shape, y[1].shape)
