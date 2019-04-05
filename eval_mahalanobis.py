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

#with open('kitti_features.pickle', 'rb') as f:
#    trainer.faster_rcnn.features = pickle.load(f)
    
with open('mahal_means.pickle', 'rb') as f:
    mahal_means = pickle.load(f)
    trainer.faster_rcnn.mahal_means = mahal_means
        
with open('mahal_cov.pickle', 'rb') as f:
    mahal_cov = pickle.load(f)
    trainer.faster_rcnn.mahal_cov = mahal_cov
    
with open('inv_mahal_cov.pickle', 'rb') as f:
    inv_mahal_cov = pickle.load(f)
    trainer.faster_rcnn.inv_mahal_cov = inv_mahal_cov

with open('kitti_labels.pickle', 'rb') as f:
    gt_labels = pickle.load(f)


test_dataset = TestDataset(opt)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=opt.num_workers)

    
with open("baseline_results.pickle", 'rb') as f:
    baseline_result = pickle.load(f)
    
print(baseline_result)

# Evaluate mahalanobis distance method
# baseline_result = eval(test_dataloader, trainer.faster_rcnn, test_num=3769)

mahal_result = eval_mahal(test_dataloader, trainer.faster_rcnn, test_num=3769)

with open("mahal_result.pickle", "wb") as f:
    pickle.dump(mahal_result, f)
    
print(mahal_result)

