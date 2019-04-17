import os
from data.voc_dataset import VOCBboxDataset
from utils.class_labels import CLASS_LABELS

def filter_voc_data_split(data_dir, split='trainval', use_difficult=True, ignore_missing_labels=True):
    """Return a new file list of files that only contain objects when taking into account use_difficult and keep_classes."""
    id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))
    ids = [id_.strip() for id_ in open(id_list_file)]
    raw_dataset = VOCBboxDataset(data_dir, split, use_difficult)

    result = []
    for id_, (_, _, label, _) in zip(ids, raw_dataset):
        if len(label) > 0:
            yield id_

def create_filtered_img_set(data_dir, split='trainval', use_difficult=True, ignore_missing_labels=True, outfile='dataset.txt'):
    filtered_voc = filter_voc_data_split(data_dir, split, use_difficult, ignore_missing_labels)

    with open(outfile, 'w') as outfile:
        for id_ in filtered_voc:
            outfile.write(id_)
            outfile.write('\n')
