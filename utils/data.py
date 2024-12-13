import torch
from torch.utils import data as data
import os
from .image_processing import imfrombytes, img2tensor
import json 
import torch.nn.functional as F


def get(filepath):
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf

def get_json(filepath):
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        jsonfile = json.load(f)
    return jsonfile


class Restoration_Dataset(data.Dataset):
    def __init__(self, config):
        super(Restoration_Dataset, self).__init__()
        
        
        self.config = config
        
        assert ('dataroot' in config) ^ ('dataroot_gt' in config and 'dataroot_lq' in config)

        if 'dataroot' in config:        
            self.gt_suffix = config['gt_suffix']
            self.lq_suffix = config['lq_suffix']
            self.gt_data_root = config['dataroot']
            self.lq_data_root = config['dataroot']

        else:   
            self.gt_suffix = ''
            self.lq_suffix = ''
            self.gt_data_root = config['dataroot_gt']
            self.lq_data_root = config['dataroot_lq']

        self.lq_paths = sorted([os.path.join(self.lq_data_root, sample) \
                                for sample in os.listdir(self.lq_data_root) \
                                if sample.endswith(self.lq_suffix + ".png")])
        
        self.gt_paths = sorted([os.path.join(self.gt_data_root, sample) \
                                for sample in os.listdir(self.gt_data_root) \
                                if sample.endswith(self.gt_suffix + ".png")])
        #Assert that all GT and LQ tuples are correctly aligned:

        assert all(["".join(gt.split("_")[0]) == "".join(lq.split("_")[0]) \
                    for (gt, lq) in zip(self.gt_paths, self.lq_paths)])
    
    def __getitem__(self, index):
        img_gt  = imfrombytes(get(self.gt_paths[index]))
        img_lq = imfrombytes(get(self.lq_paths[index]))
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': self.lq_paths[index],
            'gt_path': self.gt_paths[index]
        }
    def __len__(self):
        return len(self.lq_paths)
    



class Detection_Dataset(data.Dataset):
    def __init__(self, config):
        super(Detection_Dataset, self).__init__()
        
        
        self.config = config
        
        assert ('dataroot' in config) and ('labels_path' in config)

        self.gt_suffix = config['gt_suffix']
        self.gt_data_root = config['dataroot']
        self.labels_data_root = config['labels_path']
        

        self.gt_paths = sorted([os.path.join(self.gt_data_root, sample) \
                                for sample in os.listdir(self.gt_data_root) \
                                if sample.endswith(self.gt_suffix + ".png")], key=lambda x: int(x.split('/')[-1].split("_")[0].split(".")[0]))
        self.labels_path = sorted([os.path.join(self.labels_data_root, sample) \
                                for sample in os.listdir(self.labels_data_root)], key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #Assert that all GT and LQ tuples are correctly aligned:
        assert all(["".join(gt.split('/')[-1].split("_")[0].split(".")[0]) == "".join(label.split("/")[-1].split(".")[0]) \
                    for (gt, label) in zip(self.gt_paths, self.labels_path)])
    
    def __getitem__(self, index):
        img_gt  = imfrombytes(get(self.gt_paths[index]))
        label = get_json(self.labels_path[index])
        img_gt, _ = img2tensor([img_gt, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        
        return {
            'label': label,
            'img': img_gt,
            'label_path': self.labels_path[index],
            'img_path': self.gt_paths[index]
        }
    def __len__(self):
        return len(self.gt_paths)

class Unlabeled_Dataset(data.Dataset):
    def __init__(self, config,task):
        super(Unlabeled_Dataset, self).__init__()
        
        
        self.config = config
        self.task = task
        assert ('dataroot' in config) ^ ('dataroot_lq' in config)

        if 'dataroot' in config:        
            self.lq_suffix = config['lq_suffix']
            self.lq_data_root = config['dataroot']

        else:   
            self.lq_suffix = ''
            self.lq_data_root = config['dataroot_lq']

        self.lq_paths = sorted([os.path.join(self.lq_data_root, sample) \
                                for sample in os.listdir(self.lq_data_root) \
                                if sample.endswith(self.lq_suffix + ".png")])
        

    def __getitem__(self, index):
        img_lq = imfrombytes(get(self.lq_paths[index]))
        img_lq = img2tensor(img_lq,
                                    bgr2rgb=True,
                                    float32=True)
        
        return img_lq
    def __len__(self):
        return len(self.lq_paths)


def create_dataloader(dataset, config):
    dataloader_args = dict(config['datasets']['dataloader'])
    return torch.utils.data.DataLoader(dataset=dataset,**dataloader_args)