import os.path as osp
import warnings
import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES
from ..data_loader import CervicalDataset
@PIPELINES.register_module
class LoadImage_proposals(object):
    def __init__(self,patch_size=[1024,1024]):
        self.patch_size=patch_size

    def __call__(self, results):

        proposals = np.array(results['proposals'],dtype=np.float32)
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]
        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)

        img_key = results['img_path']
        p,img_path = img_key.split('-')
        img_path = os.path.join(results['img_path'],img_path+'.npz')
        x, y = list(map(int, p.split('_')))
        filename = img_path.split('/')[-1].split('.')[0]
        img = np.load(img_path)['img'].astype(np.float32)[x:x+self.patch_size[0],y:y+self.patch_size[1]]
        results['proposals'] = proposals
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
@PIPELINES.register_module
class LoadImage(object):
    def __init__(self):
        pass
    def __call__(self, results):
        img_path = results['img_info']
        filename = img_path.split('/')[-1].split('.')[0]
        img = np.load(img_path)['img'].astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

@PIPELINES.register_module
class LoadImage_label(object):
    def __init__(self, patch_size, to_float32=False):
        self.to_float32 = to_float32
        self.reader = CervicalDataset(patch_size)

    def __call__(self, results):
        img_path = results['img_info']
        class_id = results['class_id']

        filename = img_path.split('/')[-1].split('.')[0]
        data = self.reader.load_data(img_path,results['aug_img'],class_id)
        if data is None:
            return None
        img, bboxs = data
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        ignor = np.zeros((0, 4), dtype=np.float32)
        if bboxs.shape[0]==0:
            return None
        elif class_id is None:
            keep_inds = np.arange(bboxs.shape[0])
            if '1x' not in filename:
                ignore_class = results['ignore_class']
                keep_inds = np.empty((0),dtype=np.int)
                ignor_inds = np.empty((0),dtype=np.int)
                for i in range(1, 7):
                    if i in ignore_class:
                        ignor_inds = np.hstack((np.argwhere(bboxs[:, -1] == i)[:, 0], ignor_inds))
                    else:
                        keep_inds = np.hstack((np.argwhere(bboxs[:, -1] == i)[:, 0], keep_inds))
                if ignor_inds.shape[0]==bboxs.shape[0] or keep_inds.shape[0]==0:
                    return None
                if len(ignor_inds)>0:
                    temp_ind = np.random.randint(len(ignor_inds)+1)
                    ignor_inds = ignor_inds[:temp_ind]
                    ignor = bboxs[ignor_inds,:4].astype(np.float32).copy()

            results['gt_bboxes'] = bboxs[keep_inds,:4].astype(np.float32).copy()
            results['gt_labels'] = np.array(bboxs[keep_inds,-1], dtype=np.int64)
        else:
            keep_inds = np.argwhere(bboxs[:, -1] == class_id)[:, 0]
            results['gt_bboxes'] = bboxs[keep_inds,:4].astype(np.float32).copy()
            results['gt_labels'] = np.ones(keep_inds.shape[0], dtype=np.int64)
        results['gt_bboxes_ignore'] = ignor
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
