from __future__ import print_function, division

import numpy as np



class CervicalDataset(object):
    """Cervical dataset."""

    def __init__(self, patch_size, transform=None):

        self.patch_size = patch_size
        self.transform = transform


    def crop_patch(self, img, labels,class_id = None):
        H, W, C = img.shape
        px, py = self.patch_size
        obj_num = labels.shape[0]
        num = 0
        while True:
            num += 1
            if num>20:
                return None
            if class_id is not None:
                ll_index = np.argwhere(labels[:, -1] == class_id)[:, 0]
            # if np.random.uniform()>0.90 and len(ll_index)>0:
                index = ll_index[np.random.randint(0, len(ll_index))]
            else:
                index = np.random.randint(0, obj_num)
            ux, uy, vx, vy ,_= labels[index, :]
            if (vx-ux)>px*1.2 or (vy-uy)>py*1.2:
                continue
            else:
                break
        nx = np.random.randint(max(min(vx - px, ux, W - px), 0), min(ux + 1, W - px + 1))

        ny = np.random.randint(max(min(vy - py, uy, H - py), 0), min(uy + 1, H - py + 1))
        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        index = self.compute_overlap(patch_coord, labels, 0.5)
        index = np.squeeze(index, axis=0)

        patch = img[ny: ny + py, nx: nx + px, :]
        label = labels[index, :]
        label[:, 0] = np.maximum(label[:, 0] - patch_coord[0, 0], 0)
        label[:, 1] = np.maximum(label[:, 1] - patch_coord[0, 1], 0)
        label[:, 2] = np.minimum(self.patch_size[0] + label[:, 2] - patch_coord[0, 2], self.patch_size[0])
        label[:, 3] = np.minimum(self.patch_size[1] + label[:, 3] - patch_coord[0, 3], self.patch_size[1])
        return patch, label

    def compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index

    def load_data(self, data_path,aug_path=None,class_id = None):
        data = np.load(data_path)
        img, label = data['img'], data['label']
        img = img.astype(np.float32)
        if label is None:
            x = np.random.randint(img.shape[0]-self.patch_size[0])
            y = np.random.randint(img.shape[1]-self.patch_size[1])
            return img[x:x+self.patch_size[0],y:y+self.patch_size[1],:],np.empty((0,5))
        data = self.crop_patch(img, label,class_id)
        if data is None:
            return None

        img, label = data
        if aug_path is not None:
            try:
                aimg = np.load(aug_path)['img']
                w,h = aimg.shape[:2]
                if w>self.patch_size[0] and h>self.patch_size[1]:
                    x = np.random.randint(w - self.patch_size[0])
                    y = np.random.randint(h - self.patch_size[1])
                    aimg = aimg[x:x + self.patch_size[0], y:y + self.patch_size[1]]
                    rate = np.random.uniform(high=0.10)
                    img = img*(1-rate)+aimg*rate
            except Exception as e:
                pass
        # while True:
        #     rimg, rlabel = self.crop_patch(img, label)
        #     if rlabel.shape[0] >0 :
        #         break
        #     a = 1
        return img, label


# class WsiDataset(Dataset):
#     def __init__(self, read, y_num, x_num, strides, coordinates, patch_size, transform=None):
#         self.read = read
#         self.y_num = y_num
#         self.x_num = x_num
#         self.strieds = strides
#         self.coordinates = coordinates
#         self.patch_size = patch_size
#         self.transform = transform
#
#     def __getitem__(self, index):
#         coord_y, coord_x = self.coordinates[index]
#         img = self.read.ReadRoi(coord_x, coord_y, cfg.patch_size[0], cfg.patch_size[1], scale=20).copy()
#
#         img = img.transpose((2, 0, 1))
#         img = img.astype(np.float32) / 255.0
#
#         return torch.from_numpy(img).float(), coord_y, coord_x
#
#     def __len__(self):
#         return self.y_num * self.x_num
#
#
# def collater(data):
#     imgs = [s['img'] for s in data]
#     labels = [s['label'] for s in data]
#
#     imgs = torch.stack(imgs, dim=0)
#     imgs = imgs.permute((0, 3, 1, 2))
#
#     max_num_labels = max(label.shape[0] for label in labels)
#
#     if max_num_labels > 0:
#         label_pad = torch.ones((len(labels), max_num_labels, 4)) * -1
#
#         for idx, label in enumerate(labels):
#             if label.shape[0] > 0:
#                 label_pad[idx, :label.shape[0], :] = label
#     else:
#         label_pad = torch.ones((len(labels), 1, 4)) * -1
#
#     return {'img': imgs, 'label': label_pad}
