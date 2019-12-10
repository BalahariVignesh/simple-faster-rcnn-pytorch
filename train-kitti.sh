#!/bin/bash

if [[ $# -gt 0 && "$1" == "visdom" ]]; then
    nohup python -m visdom.server &
fi

#sensible-browser http://localhost:8097

python -W once train.py train --env='fasterrcnn-caffe' --voc-data-dir=/media/tadenoud/DATADisk/datasets/kitti_2d/VOC2012
