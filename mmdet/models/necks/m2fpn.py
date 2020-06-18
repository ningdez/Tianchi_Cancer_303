'''
This code is based on pytorch_ssd and RFBNet.
Details about the modules:
               TUM - Thinned U-shaped Module
               MLFPN - Multi-Level Feature Pyramid Network
               M2Det - Multi-level Multi-scale single-shot object Detector

Author:  Qijie Zhao (zhaoqijie@pku.edu.cn)
Finished Date:  01/17/2019

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import warnings
warnings.filterwarnings('ignore')
from ..registry import NECKS
from ..utils import ConvModule

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6,
                 conv_cfg=None,
                 norm_cfg=None
    ):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), ConvModule(self.in1, self.planes, 3, 2, 1,conv_cfg=conv_cfg,norm_cfg=norm_cfg))
        for i in range(self.scales - 2):
            if not i == self.scales - 3:
                self.layers.add_module(
                    '{}'.format(len(self.layers)),
                    ConvModule(self.planes, self.planes, 3, 2, 1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
                )
            else:
                self.layers.add_module(
                    '{}'.format(len(self.layers)),
                    ConvModule(self.planes, self.planes, 3, 1, 0,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
                )
        self.toplayer = nn.Sequential(ConvModule(self.planes, self.planes, 1, 1, 0,conv_cfg=conv_cfg,norm_cfg=norm_cfg))

        self.latlayer = nn.Sequential()
        for i in range(self.scales - 2):
            self.latlayer.add_module(
                '{}'.format(len(self.latlayer)),
                ConvModule(self.planes, self.planes, 3, 1, 1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
            )
        self.latlayer.add_module('{}'.format(len(self.latlayer)), ConvModule(self.in1, self.planes, 3, 1, 1,conv_cfg=conv_cfg,norm_cfg=norm_cfg))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales - 1):
                smooth.append(
                    ConvModule(self.planes, self.planes, 1, 1, 0,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
                )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _, _, H, W = y.size()
        if fuse_type == 'interp':
            return F.interpolate(x, size=(H, W), mode='nearest') + y
        else:
            raise NotImplementedError
            # return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x, y], 1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                self._upsample_add(
                    deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers) - 1 - i])
                )
            )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                    self.smooth[i](deconved_feat[i + 1])
                )
            return smoothed_feat
        return deconved_feat

class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels,
                                            self.planes * self.num_levels // 16,
                                            1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels // 16,
                                            self.planes * self.num_levels,
                                            1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf * _tmp_f)
        return attention_feat

@NECKS.register_module
class M2FPN(nn.Module):
    def __init__(self,
                 num_levels = 8,
                 num_scales = 5,
                 sfam=False,
                 smooth=True,
                 in_channels = [512,2048],
                 out_channels=256, conv_cfg=None,
                 norm_cfg=None):
        '''
        M2Det: Multi-level Multi-scale single-shot object Detector
        '''
        super(M2FPN,self).__init__()
        self.planes = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.sfam = sfam
        self.smooth = smooth
        self.in_channels = in_channels
        self.shallow_out =256
        self.deep_out =512
        self.construct_modules()

    def construct_modules(self,):
        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=True, 
                            input_planes=self.planes//2, 
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512)) #side channel isn't fixed.
            else:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=False, 
                            input_planes=self.planes//2, 
                            is_smooth=self.smooth, 
                            scales=self.num_scales,
                            side_channel=self.planes))

        self.reduce= ConvModule(self.in_channels[0], self.shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce_1= ConvModule(self.in_channels[2], self.in_channels[1], kernel_size=1, stride=1)
        self.up_reduce_2= ConvModule(self.in_channels[1], self.deep_out, kernel_size=1, stride=1)

        self.Norm = nn.BatchNorm2d(256*8)
        self.leach = nn.ModuleList([ConvModule(
                    self.deep_out+self.shallow_out,
                    self.planes//2,
                    kernel_size=(1,1),stride=(1,1))]*self.num_levels)

        # construct localization and recognition layers
        conv_out = nn.ModuleList()
        for i in range(self.num_scales):
            conv_out.append(nn.Conv2d(self.planes*self.num_levels,
                                       self.planes,
                                       3, 1, 1))
        self.conv_out = nn.ModuleList(conv_out)

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self,x):
        assert len(x)==len(self.in_channels)
        # loc,conf = list(),list()
        # base_feats = list()
        # if 'vgg' in self.net_family:
        #     for k in range(len(self.base)):
        #         x = self.base[k](x)
        #         if k in self.base_out:
        #             base_feats.append(x)
        # elif 'res' in self.net_family:
        #     base_feats = self.base(x, self.base_out)
        up_feats = x[1] + F.interpolate(self.up_reduce_1(x[2]),scale_factor=2,mode='nearest')
        base_feature = torch.cat(
                (self.reduce(x[0]), F.interpolate(self.up_reduce_2(up_feats),scale_factor=2,mode='nearest')),1
                )

        # tum_outs is the multi-level multi-scale feature
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature), 'none')]
        for i in range(1,self.num_levels,1):
            tum_outs.append(
                    getattr(self, 'unet{}'.format(i+1))(
                        self.leach[i](base_feature), tum_outs[i-1][-1]
                        )
                    )
        # concat with same scales
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs],1) for i in range(self.num_scales, 0, -1)]
        
        # forward_sfam
        if self.sfam:
            sources = self.sfam_module(sources)
        sources[0] = self.Norm(sources[0])
        output = []
        for (x,cout) in zip(sources, self.conv_out):
            output.append(cout(x))

        return tuple(output)
