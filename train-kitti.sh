# nohup python -m visdom.server &

#sensible-browser http://localhost:8097

python train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain --voc-data-dir=/media/tadenoud/DATADisk/datasets/kitti_2d/VOC2012
