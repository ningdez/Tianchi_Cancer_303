import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from torch.utils.checkpoint import checkpoint

class BiFPN_Block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 end_block=True,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 use_weight=False):
        super(BiFPN_Block, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.end_block = end_block
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.use_weight=use_weight

        self.td_up_convs = nn.ModuleList()
        self.td_dw_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.td_mid_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_mid_convs = nn.ModuleList()
        if self.use_weight:
            self.td_weight = nn.Parameter(torch.rand((self.backbone_end_level-self.start_level,2),dtype=torch.float,requires_grad=True))
            self.up_weight = nn.Parameter(torch.rand((self.backbone_end_level-self.start_level,3),dtype=torch.float,requires_grad=True))

        for i in range(self.start_level, self.backbone_end_level):
            if i > self.start_level and i < self.backbone_end_level - 1:

                td_mid_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    activation=self.activation,
                    inplace=False)
                self.td_mid_convs.append(td_mid_conv)

            if i > self.start_level:
                td_up_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    activation=self.activation,
                    inplace=False)
                self.td_up_convs.append(td_up_conv)
            fpn_mid_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.fpn_mid_convs.append(fpn_mid_conv)

            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)

            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def Resize(self, input, scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample(input)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build td
        if self.use_weight:
            td_weight = torch.softmax(self.td_weight,dim=1)
            up_weight = torch.softmax(self.up_weight,dim=1)

        tds = [td_up_conv(inputs[i+1+ self.start_level])for i,td_up_conv in enumerate(self.td_up_convs)]
        for i in range(len(tds)-2, -1, -1):
            w1,w2=1,1
            if self.use_weight:
                assert i<td_weight.shape[0]
                w1, w2 = td_weight[i]
            tds[i] = self.td_mid_convs[i](w1*tds[i]+ w2*F.interpolate(
                tds[i+1], size=tds[i].size()[2:], mode='bilinear'))
        tds = tds[0:-1]
        # tds = [td_dw_conv(tds[i]) for i, td_dw_conv in enumerate(self.td_dw_convs)]
        # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build down-top path
        used_backbone_levels = len(laterals)
        w1, w2 = 1, 1
        if self.use_weight:
            w1, w2 = td_weight[-1]
        laterals[0] = self.fpn_mid_convs[0](w1*laterals[0] + w2* F.interpolate(tds[0], size=laterals[0].size()[2:], mode='bilinear'))
        for i in range(1, used_backbone_levels - 1):
            w1, w2, w3 = 1, 1, 1
            if self.use_weight:
                assert i-1<up_weight.shape[0]
                w1, w2, w3 = up_weight[i-1]
            laterals[i] = self.fpn_mid_convs[i](w1*laterals[i] + w2*tds[i - 1] + w3*F.max_pool2d(laterals[i - 1], 1, stride=2))

        # build outputs
        # part 1: from original levels

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs) and self.end_block:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        elif len(outs) != len(self.in_channels):
            outs = [inputs[0]]+outs
            # print(len(outs))
        return outs


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 num_stages=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 use_weight=False,
                 out_stages=[-1]):
        super(BiFPN, self).__init__()
        self.block_list = nn.ModuleList()
        assert len(out_stages)==1 or len(out_stages)==2
        self.out_stages = out_stages
        self.num_stages = num_stages
        if len(out_stages) == 1:
            self.out_stages = [self.num_stages-1]
        assert self.out_stages[-1]==self.num_stages-1
        for i in range(self.num_stages):
            end_block = True if i in self.out_stages else False
            out_channel_num =  4
            in_channels = [out_channels] * out_channel_num if i>0 else in_channels
            block = BiFPN_Block(
                in_channels=in_channels,
                num_outs=num_outs,
                out_channels=out_channels,
                start_level=start_level,
                end_level=end_level,
                end_block=end_block,
                add_extra_convs=add_extra_convs,
                extra_convs_on_inputs=extra_convs_on_inputs,
                relu_before_extra_convs=relu_before_extra_convs,
                no_norm_on_lateral=no_norm_on_lateral,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                use_weight=use_weight)
            self.block_list.append(block)
            # setattr(self,str(i), self.block_list[i])
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for i, block in enumerate(self.block_list):
            block.init_weights()
    def forward(self, input):
        out = []
        # for i in range(self.num_stages):
        #     input = getattr(self,str(i))(input[:4])
        #     if i in self.out_stages:
        #         out+=input
        for i,block in enumerate(self.block_list):
            input = block(input[:4])
            # if i in self.out_stages:
            #     out+=input
        # print(len(out))
        return input
        # return tuple(out)