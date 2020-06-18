import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .reppoints_head import RepPointsHead


@HEADS.register_module
class REPRPNHead(RepPointsHead):
    """Guided-Anchor-based RPN head."""

    def __init__(self, in_channels, **kwargs):
        super(REPRPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(REPRPNHead, self)._init_layers()

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        super(REPRPNHead, self).init_weights()

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        (cls_out, pts_out_init, pts_out_refine) = super(REPRPNHead, self).forward_single(x)
        return cls_out, pts_out_init, pts_out_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        losses = super(REPRPNHead, self).loss(
            cls_scores,
            pts_preds_init,
            pts_preds_refine,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore)
        return dict(
            loss_cls=losses['loss_cls'],
            loss_pts_init=losses['loss_pts_init'],
            loss_pts_refine=losses['loss_pts_refine'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          use_nms=True):
        mlvl_proposals = []
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = torch.stack([x1, y1, x2, y2], dim=-1)
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]


            proposals = torch.cat([proposals, scores], dim=-1)
            # print('1', proposals.size(1))
            # NMS in current level
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        # print('2', proposals.size(1))
        if cfg.nms_across_levels:
            # NMS across multi levels
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]

        return proposals
