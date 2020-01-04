import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


def parse_annotations_xml(f):
    root = ET.parse(f).getroot()
       
    size = root.find('size')
    size = {
        "width": size.find("width").text,
        "height": size.find("height").text,
        "depth": size.find("depth").text
    }
    
    objects = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        objects.append({
            "name": obj.find('name').text,
            "xmin": int(bndbox.find('xmin').text),
            "xmax": int(bndbox.find('xmax').text),
            "ymin": int(bndbox.find('ymin').text),
            "ymax": int(bndbox.find('ymax').text) 
        })
        
    return {
        "filename": root.find('filename').text,
        "folder": root.find('folder').text,
        "size": size,
        "objects": objects
    }

def get_annotations(f):
    bboxes = []
    labels = []
    
    annotations = parse_annotations_xml(f)
    for b in annotations['objects']:
        bboxes.append([b['ymin'], b['xmin'], b['ymax'], b['xmax']])
        labels.append(b['name'])
    
    return np.array(bboxes).astype(np.float32), np.array(labels).astype(np.str)


def percentage_boxes_to_img_dim_boxes(image_np, detection_boxes):
    # Convert percentage boxes to image dimensions
    img_height, img_width, img_depth = image_np.shape
    detection_boxes[:,(0,2)] *= img_height
    detection_boxes[:,(1,3)] *= img_width
    return detection_boxes


def img_dim_boxes_to_percentage_boxes(image_np, detection_boxes):
    # Convert percentage boxes to image dimensions
    img_height, img_width, img_depth = image_np.shape
    detection_boxes[:,(0,2)] /= img_height
    detection_boxes[:,(1,3)] /= img_width
    return detection_boxes


_idd_labels_text_to_id = {
    'car': 0,
    'person': 1,
    'bicycle': 2,
    'bus': 3,
    'traffic sign': 4,
    'train': 5,
    'motorcycle': 6,
    'traffic light': 7,
    'vehicle fallback': 8,
    'truck': 9,
    'autorickshaw': 10,
    'animal': 11,
    'caravan': 12,
    'rider': 13,
    'trailer': 14 
}

_idd_labels_id_to_text = {v: k for k, v in _idd_labels_text_to_id.items()}


def idd_labels_str_to_int(labels):
    return [_idd_labels_text_to_id[l] for l in labels]


def idd_labels_int_to_str(labels):
    return [_idd_labels_id_to_text[l] for l in labels]


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
    

class IndiaDrivingDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, keep_labels=None):
        super().__init__()
        self.root_dir = os.path.realpath(root_dir)
        self.split = split + ".txt"
        self.transform = transform
        self.keep_labels = keep_labels
        
        assert(os.path.exists(os.path.join(self.root_dir, self.split)))
        
        with open(os.path.join(self.root_dir, self.split), 'r') as f:
               self.file_list = f.readlines()
        
        if self.keep_labels is not None:
            self._filter_file_list()
        
    def _filter_file_list(self):
        result = []
        for idx in range(len(self.file_list)):
            label_name = os.path.join(self.root_dir,'Annotations', self.file_list[idx].strip() + '.xml')
            _, labels = get_annotations(label_name)
            labels = np.array(idd_labels_str_to_int(labels)).astype(np.int)
            
            if len(set(labels).intersection(set(self.keep_labels))) > 0:
                result.append(self.file_list[idx].strip())
                
        self.file_list = result
    
    def __len__(self):
        return len(self.file_list)
           
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,'JPEGImages', self.file_list[idx].strip() + '.jpg')
        img = read_image(img_name)
        
        label_name = os.path.join(self.root_dir,'Annotations', self.file_list[idx].strip() + '.xml')
        bboxes, labels = get_annotations(label_name)
        labels = np.array(idd_labels_str_to_int(labels)).astype(np.int)

        if self.transform:
            img = self.transform(img)

        return img, bboxes, labels
