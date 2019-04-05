import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import numpy as np
import pickle

from data.dataset import Dataset, TestDataset
from torch.utils.data import DataLoader
from train import eval, eval_mahal

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('./checkpoints/fasterrcnn_02272314_0.6720744290992889')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model

dataset = Dataset(opt)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False, \
                        # pin_memory=True,
                        num_workers=opt.num_workers)


mahal_means, mahal_cov = trainer.faster_rcnn.train_ood(dataloader)

with open('kitti_features.pickle', 'wb') as f:
    pickle.dump(trainer.faster_rcnn.features, f)
    
with open('mahal_means.pickle', 'wb') as f:
    pickle.dump(mahal_means, f)

with open('mahal_cov.pickle', 'wb') as f:
    pickle.dump(mahal_cov, f)

with open('inv_mahal_cov.pickle', 'wb') as f:
    pickle.dump(trainer.faster_rcnn.inv_mahal_cov, f)

