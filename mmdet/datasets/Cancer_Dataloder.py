import os
from glob import glob
from .registry import DATASETS
from .pipelines import Compose
from torch.utils.data import Dataset
import numpy as np
import json
@DATASETS.register_module
class CancerDataset(Dataset):
    CLASSES = ('bad')
    def __init__(self,data_path,pipeline,test_mode=False,neg_path=None,multi_scales=['1x'],
                 aug_path=None,ignore_class=[5],
                 ignore_image=['233','2687','4110','8515','8635','9636','6397','5062'],
                 single_label = None):
        self.data_path = data_path
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.ignore_class = ignore_class
        self.single_label = single_label
        if self.single_label is not None:
            print(self.single_label)
            json_file = os.path.join(data_path,self.single_label+'.json')
            with open(json_file, 'r') as f:
                img_infos = json.loads(f.read())
        else:
            img_infos = glob(os.path.join(data_path, '*.npz'))
        self.img_ids = []
        if neg_path is not None:
            img_infos += glob(os.path.join(neg_path, '*.npz'))
        self.img_info = []
        print('multi_scales', multi_scales)
        for multi_scale in multi_scales:
            if self.test_mode :
                self.img_info += [info for info in img_infos if multi_scale in info and (multi_scale =='1x' or info.split('/')[-1].split('_')[0] not in ignore_image)]
            else:
                self.img_info += [info for info in img_infos if multi_scale in info]
        for img_info in self.img_info:
            self.img_ids.append(img_info.split('/')[-1].split('.')[0])
        double_file = os.path.join(data_path, 'double_file.json')
        if os.path.exists(double_file) and not self.test_mode:
            with open(double_file, 'r') as f:
                dfile_name = json.loads(f.read())
            self.double_idx = []
            double_name = []
            for idx,img_info in enumerate(self.img_info):
                img_name = img_info.split('/')[-1].split('_')[0]
                if img_name in dfile_name:
                    double_name.append(img_info)
                    self.double_idx.append(idx)
                # if '1-5x' in img_info:
                #     double_name.append(img_info)
            self.img_info+=double_name
        else:
            self.double_idx = np.arange(len(self.img_info))
        self.flag = np.zeros(len(self.img_info), dtype=np.uint8)
        self.aug_img = [None]
        if aug_path is not None:
            self.aug_img = glob(os.path.join(aug_path, '*.npz'))



        print('file num',len(self.img_info))
        print('double num',len(self.double_idx))
        self.cat_ids = {0:"ASC-H",1:"ASC-US",2:"HSIL",3:"LSIL",4:"Trichomonas",5:"Candida"}
        self.class_dict = {"ASC-H": 1, "ASC-US": 2, "HSIL": 3, "LSIL": 4, "Trichomonas": 5, "Candida": 6}
    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = np.random.randint(len(self.img_info))
                # idx = self.double_idx[didx]
                continue
            return data
        # if self.test_mode:
        #     return self.prepare_test_img(idx)
        # while True:
        #     data = self.prepare_train_img(idx)
        #     if data is None:
        #         idx = self._rand_another(idx)
        #         continue
        #     return data

    def prepare_train_img(self, idx):
        # if idx>len(self.img_info):
        #     didx = np.random.randint(len(self.double_idx))
        #     idx = self.double_idx[didx]
        # print(idx)
        img_info = self.img_info[idx]
        results = dict(img_info = img_info,aug_img=self.aug_img[np.random.randint(len(self.aug_img))],ignore_class=self.ignore_class,
                       class_id=None if self.single_label is None else self.class_dict[self.single_label],)

        return self.pipeline(results)

