from torchvision import models
from .gcn_parts import *
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, n_classes, resnet_depth: int=50):
        super(GCN, self).__init__()

        self.num_classes = n_classes
        self.num_int_channels = 19

        if resnet_depth == 101:
            resnet = models.resnext101_32x8d(pretrained=True)
        elif resnet_depth == 50:
            resnet = models.resnext50_32x4d(pretrained=True)
        else:
            raise NotImplementedError

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN_module(2048, self.num_int_channels, ks=5)
        self.gcn2 = GCN_module(1024, self.num_int_channels, ks=5)
        self.gcn3 = GCN_module(512, self.num_int_channels, ks=5)
        self.gcn4 = GCN_module(256, self.num_int_channels, ks=5)

        self.refine1 = Refine(self.num_int_channels)
        self.refine2 = Refine(self.num_int_channels)
        self.refine3 = Refine(self.num_int_channels)
        self.refine4 = Refine(self.num_int_channels)
        self.refine5 = Refine(self.num_int_channels)
        self.refine6 = Refine(self.num_int_channels)
        self.refine7 = Refine(self.num_int_channels)
        self.refine8 = Refine(self.num_classes)
        self.refine9 = Refine(self.num_classes)

        self.tconv1 = UpscalingConv2d(self.num_int_channels, self.num_int_channels)
        self.tconv2 = UpscalingConv2d(self.num_int_channels, self.num_int_channels)
        self.tconv3 = UpscalingConv2d(self.num_int_channels, self.num_int_channels)
        self.tconv4 = UpscalingConv2d(self.num_int_channels, self.num_classes)
        self.tconv5 = UpscalingConv2d(self.num_classes, self.num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer0(x)

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(fm1))

        fs1 = self.refine5(self.tconv1(gcfm1) + gcfm2)
        fs2 = self.refine6(self.tconv2(fs1) + gcfm3)
        fs3 = self.refine7(self.tconv3(fs2) + gcfm4)
        out1 = self.refine8(self.tconv4(fs3))
        out = self.refine9(self.tconv5(out1))

        out = self.sigmoid(out)

        return {'out': out, 'fs3': fs3, 'fs2': fs2, 'fs1': fs1, 'gcmf1': gcfm1}
