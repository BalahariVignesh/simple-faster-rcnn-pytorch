from pprint import pprint

from .class_labels import CLASS_LABELS
# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = "/home/tadenoud/Documents/kitti/VOC2012/"
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 0
    test_num_workers = 0

    num_fg_classes = len(CLASS_LABELS)
    
    # Don't care class present (as in kitti data)
    # Should not be counted in num_fg_classes and be listed as the final label in VOC_BBOX_LABEL_NAMES
    # Will treat all objects not present in VOC_BBOX_LABEL_NAMES as unlabeled and not backpropagate errors from these classes
    dont_care_class = True

    ignore_missing_labels = True

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    # train_num = 3712  # all classes
    # test_num = 3769  # all classes
    # train_num = 3429  # cars only
    # test_num = 3578  # cars only
    # train_num = 3452  # cars and vans
    # test_num = 3581  # cars and vans
    train_num = 3012  # pedestrians and cyclists
    test_num = 3185  # pedestrians and cyclists

    
    # model
    load_path = None

    caffe_pretrain = True # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
