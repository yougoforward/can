from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['new_can4', 'get_new_can4']

class new_can4(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, atrous_rates=(12, 24, 36), decoder=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_can4, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = new_can4Head(2048, nclass, norm_layer, self._up_kwargs, atrous_rates)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c1,c2,c3,c4)
        x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class new_can4Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates):
        super(new_can4Head, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        
        self.block = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(2*inter_channels, out_channels, 1))

        self.fpn_head = fcn_fpnHead(in_channels, norm_layer, self._up_kwargs)
        self.aspp = ASPP_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4):
        x = self.fpn_head(c1,c2,c3,c4)
        #dual path
        out = self.aspp(x)
        pred = self.block(out)

        return pred
# def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
#     block = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
#                   dilation=atrous_rate, bias=False),
#         norm_layer(out_channels),
#         nn.ReLU(True))
#     return block
class fcn_fpnHead(nn.Module):
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(fcn_fpnHead, self).__init__()
        # inter_channels = in_channels//4
        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(),
        #                            )

        # self.localUp2=localUp(256, in_channels, norm_layer, up_kwargs)
        self.localUp3=localUp(512, in_channels, norm_layer, up_kwargs)
        self.localUp4=localUp(1024, in_channels, norm_layer, up_kwargs)

    def forward(self, c1,c2,c3,c4):
        out = self.localUp4(c3, c4)
        out = self.localUp3(c2, out)
        # out = self.localUp2(c1, out)
        # out = self.conv5(c4)
        return out

class localUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(localUp, self).__init__()
        self.connect = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU())
        self._up_kwargs = up_kwargs



    def forward(self, c1,c2):
        n,c,h,w =c1.size()
        c1 = self.connect(c1) # n, 64, h, w
        c2 = F.interpolate(c2, (h,w), **self._up_kwargs)
        out = c1+c2
        return out


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = inter_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.b01 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b11 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b21 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b31 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        # self.project = nn.Sequential(
        #     nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        self.project = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        
    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        out = self.project(torch.cat((feat0, feat1, feat2, feat3), 1))
        out = torch.cat([out, feat4], dim=1)
        return out

def get_new_can4(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = new_can4(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
