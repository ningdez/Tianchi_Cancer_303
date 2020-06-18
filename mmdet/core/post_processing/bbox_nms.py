import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   KL_loss=False,
                   vote=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        _num = 8 if KL_loss else 4
        if multi_bboxes.shape[1] == _num:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * _num:(i + 1) * _num]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        # cls_dets = torch.cat([_bboxes[:,:4], _scores[:, None],torch.exp(_bboxes[:,4:])], dim=1)
        if not KL_loss or not vote:
            cls_dets = torch.cat([_bboxes[:, :4], _scores[:, None]], dim=1)
            cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        else:
            cls_dets = torch.cat([_bboxes[:, :4], _scores[:, None], torch.exp(torch.clamp(_bboxes[:, 4:],min=-4))], dim=1)
            cls_dets = nms_class(cls_dets,nms_type=nms_type,score_thr=score_thr, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, 4].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        _num = 9 if KL_loss else 5
        bboxes = multi_bboxes.new_zeros((0, _num))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels

def nms_class(clsboxes,nms_type='soft_nms',sigma=0.5,iou_thr=0.5,score_thr=0.05,iou_sigma=0.02):
    assert clsboxes.shape[1] == 5 or clsboxes.shape[1] == 9
    keep = []
    while clsboxes.shape[0] > 0:
        maxidx = torch.argmax(clsboxes[:, 4])
        maxbox = clsboxes[maxidx].unsqueeze(0)
        clsboxes = torch.cat((clsboxes[:maxidx], clsboxes[maxidx + 1:]), 0)
        iou = iou_calc3(maxbox[:, :4], clsboxes[:, :4])
        # KL VOTE

        ioumask = iou > 0
        klbox = clsboxes[ioumask]
        klbox = torch.cat((klbox, maxbox), 0)
        kliou = iou[ioumask]
        klvar = klbox[:, -4:]
        pi = torch.exp(-1 * torch.pow((1 - kliou), 2) / iou_sigma)
        pi = torch.cat((pi, torch.ones(1).cuda()), 0).unsqueeze(1)
        pi = pi / klvar
        pi = pi / pi.sum(0)
        maxbox[0, :4] = (pi * klbox[:, :4]).sum(0)
        keep.append(maxbox)

        weight = torch.ones_like(iou)
        if nms_type!='soft_nms':
            weight[iou > iou_thr] = 0
        else:
            weight = torch.exp(-1.0 * (iou ** 2 / sigma))
        clsboxes[:, 4] = clsboxes[:, 4] * weight
        filter_idx = (clsboxes[:, 4] >= score_thr).nonzero().squeeze(-1)
        clsboxes = clsboxes[filter_idx]
    return torch.cat(keep, 0).to(clsboxes.device)


def iou_calc3(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

