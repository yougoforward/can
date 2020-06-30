from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['new_can3', 'get_new_can3']

class new_can3(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, atrous_rates=(12, 24, 36), decoder=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_can3, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = new_can3Head(2048, nclass, norm_layer, self._up_kwargs,atrous_rates)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)

        outputs = []
        x, coarse, free = self.head(c4, c1)
        x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        coarse = F.interpolate(coarse, (h,w), **self._up_kwargs)
        outputs.append(coarse)
        free = F.interpolate(free, (h,w), **self._up_kwargs)
        outputs.append(free)
        
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class new_can3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates):
        super(new_can3Head, self).__init__()
        inter_channels = in_channels // 4
        self.aspp = ASPP_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)

        self._up_kwargs = up_kwargs
        
        self.block1 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))
        self.block2 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(2*inter_channels, out_channels, 1))
        self.block3 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, xl):
        n,c,h,w = xl.size()
        #dual path
        aspp1, aspp2, out = self.aspp(x)


        #context sensitive
        # coarse = self.block1(aspp1)
        pred = self.block2(out)

        #context free
        context_free = self.block3(aspp2)
        return pred, context_free

# def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
#     block = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
#                   dilation=atrous_rate, bias=False),
#         norm_layer(out_channels),
#         nn.ReLU(True))
#     return block

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
        self.project1 = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.project2 = nn.Sequential(
            nn.Conv2d(4*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.context_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.Sigmoid())
        self.project = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        # feat01 = self.b01(x)
        # feat11 = self.b11(x)
        # feat21 = self.b21(x)
        # feat31 = self.b31(x)
        y1 = self.project1(torch.cat((feat0, feat1, feat2, feat3), 1))
        y2 = self.project2(torch.cat((feat0, feat1, feat2, feat3), 1))

        # y2 = self.project2(torch.cat((feat01, feat11, feat21, feat31), 1))
        # att = self.context_att(torch.cat([y1, y2], dim=1))
        # att_list = torch.split(att, 1, dim=1)
        # out = self.project(torch.cat([y1*att_list[0], y2*att_list[1]], dim=1))
        att = self.context_att(y2)
        out = y1+y1*att
        out = torch.cat([out, feat4], dim=1)
        return y1,y2,out

def get_new_can3(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = new_can3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
