from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.xcorr import xcorr_pixelwise
from pysot.models.attention.attention import SEModule,NONLocalBlock2D

class IRCA(nn.Module):
    def __init__(self):
        super(IRCA, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

def con_bn_ru(in_channels,ker_size,pad=[0,0]):
    return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(ker_size[0],ker_size[1]),padding=(pad[0],pad[1]), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

class MultiIRCA(IRCA):
    def __init__(self,cls_out_channels):
        super(MultiIRCA, self).__init__()
        self.weight = nn.Parameter(0.33 * torch.ones(3))
        self.adjust = nn.Parameter(0.1*torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
        self.channelatt=SEModule(49*3,3)
        self.spatialatt=NONLocalBlock2D(in_channels=49*3,inter_channels=49)
        self.hide=49*3
        self.region11 = nn.Sequential(
            con_bn_ru(self.hide,[3,3]),
            con_bn_ru(self.hide,[3,3]),
            con_bn_ru(self.hide,[3,3])
        )
        self.region21 = nn.Sequential(
            con_bn_ru(self.hide, [5,3],[1,0]),
            con_bn_ru(self.hide, [5,3],[1,0]),
            con_bn_ru(self.hide, [5,3],[1,0])
        )
        self.region12 = nn.Sequential(
            con_bn_ru(self.hide, [3, 5], [0, 1]),
            con_bn_ru(self.hide, [3, 5], [0, 1]),
            con_bn_ru(self.hide, [3, 5], [0, 1])
        )


        self.clshead = nn.Sequential(
            con_bn_ru(self.hide, [3, 3], [1, 1]),
            nn.Conv2d(self.hide,cls_out_channels , kernel_size=3,padding=1)
        )

        self.reghead = nn.Sequential(
            con_bn_ru(self.hide,[3,3], [1,1]),
            nn.Conv2d(self.hide, 4, kernel_size=3,padding=1)
        )


    def forward(self, z_fs, x_fs):
        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s
        weight = F.softmax(self.weight, 0)
        features = xcorr_pixelwise(x_fs[0], z_fs[0])
        for i in range(len(x_fs) - 1):
            features_new = xcorr_pixelwise(x_fs[i + 1], z_fs[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.channelatt(features)
        features = self.spatialatt(features)
        mfeature = [self.region11(features),
                    self.region21(features),
                    self.region12(features)]
        mfeature=weighted_avg(mfeature,weight)
        cls=self.clshead(mfeature)
        loc = self.reghead(mfeature)
        loc=torch.exp(loc*self.adjust+self.bias)
        return cls, loc
