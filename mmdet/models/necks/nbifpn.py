from ..registry import NECKS
import torch.nn as nn
from .nas_fpn import GPCell,SumCell
from .bifpn import BiFPN
from mmcv.cnn import caffe2_xavier_init
@NECKS.register_module
class NBiFPN(nn.Module):
    def __init__(self,out_channels,stack_times,norm_cfg=None, **kwargs):
        super(NBiFPN,self).__init__()
        self.bifpn = BiFPN(out_channels=out_channels,norm_cfg=norm_cfg, **kwargs)
        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()
        for _ in range(stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GPCell(out_channels, norm_cfg=norm_cfg)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GPCell(with_conv=False)
            stage['sum_55_5'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GPCell(with_conv=False)
            stage['sum_77_7'] = SumCell(out_channels, norm_cfg=norm_cfg)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GPCell(out_channels, norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)
        self.bifpn.init_weights()
    def forward(self, input):
        feats = self.bifpn(input)
        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return p3, p4, p5, p6, p7