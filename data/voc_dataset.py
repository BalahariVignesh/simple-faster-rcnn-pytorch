from __future__ import  absolute_import

import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image
from utils.config import opt
from utils.class_labels import CLASS_LABELS


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False, img_type='png'
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        with open(id_list_file) as f:
            self.ids = [id_.strip() for id_ in f]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = CLASS_LABELS
        self.img_type = img_type

        self.filter_ids()

    def filter_ids(self):
        """Remove images from dataset if they contain no FG objects."""
        result = []
        for id_ in self.ids:
            anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
            label = list()
            for obj in anno.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name in CLASS_LABELS:
                    label.append(CLASS_LABELS.index(name))

            if len(label) > 0:
                result.append(id_)
        
        self.ids = result

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            name = obj.find('name').text.lower().strip()
            #print(name) 
            if name not in CLASS_LABELS:
                if opt.dont_care_class and name == 'dontcare':
                    label.append(-1)  # All -1 gt_labels should not be backpropagated during training
                elif opt.ignore_missing_labels:
                    # print("ignoring", name)
                    continue
            else:
                label.append(CLASS_LABELS.index(name))

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(float(bndbox_anno.find(tag).text)) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()

        if len(label):
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            # When `use_difficult==False`, all elements in `difficult` are False.
            difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
        else:
            bbox = np.empty((0,4))
            label = np.empty((0,))            
            difficult = np.empty((0,))

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.' + self.img_type)
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example
