#!/bin/bash

python -W once eval.py train --env='fasterrcnn-caffe' --voc-data-dir=/media/tadenoud/DATADisk/datasets/kitti_2d/VOC2012 --load-path=./checkpoints/cars_pedestrians/fasterrcnn_12111330_0.5222126970906915
