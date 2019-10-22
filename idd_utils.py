import xml.etree.ElementTree as ET
import numpy as np


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
    
    return np.array(bboxes), np.array(labels)