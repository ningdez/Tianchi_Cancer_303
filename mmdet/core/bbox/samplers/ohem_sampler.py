import torch
import numpy as np
from ..transforms import bbox2roi
from .base_sampler import BaseSampler


class OHEMSampler(BaseSampler):
    """
    Online Hard Example Mining Sampler described in [1]_.

    References:
        .. [1] https://arxiv.org/pdf/1604.03540.pdf
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 expected_rate=2,
                 **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                          add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]
        self.expected_rate = expected_rate
        self.current_stage = context.current_stage
    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            bbox_feats = self.bbox_roi_extractor(
                feats[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, _ = self.bbox_head(bbox_feats)
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            # num = num_expected if len(inds)<num_expected*2 else num_expected*2
            _, topk_loss_inds = loss.topk(min(len(inds),int(num_expected*self.expected_rate)))
        return inds[self.random_choice(topk_loss_inds, num_expected)]
        # return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        # print(pos_inds.numel(),self.current_stage)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        elif num_expected * self.expected_rate < pos_inds.numel():
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard negative samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        # num_expected = min(num_expected,max(pos_inds.numel()*3,48))
        if len(neg_inds) <= num_expected:
            return neg_inds
        elif num_expected*self.expected_rate < len(neg_inds):
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    assign_result.labels[neg_inds], feats)
        else:
            return self.random_choice(neg_inds, num_expected)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]