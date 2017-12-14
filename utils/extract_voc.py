"""
Extract certain types of image and annotation file from VOC PASCAL 2007 and 2012
"""

import os
import xml.etree.ElementTree as ET
from shutil import copyfile

VOC_LABELS = (
    'aeroplane',  # 0
    'bicycle',  # 1
    'bird',  # 2
    'boat',  # 3
    'bottle',  # 4
    'bus',  # 5
    'car',  # 6
    'cat',  # 7
    'chair',  # 8
    'cow',  # 9
    'diningtable',  # 10
    'dog',  # 11
    'horse',  # 12
    'motorbike',  # 13
    'person',  # 14
    'pottedplant',  # 15
    'sheep',  # 16
    'sofa',  # 17
    'train',  # 18
    'tvmonitor',  # 19
)

xml_dir = '/data/datasets/yanfu/VOCdevkit/VOC2012/Annotations'
img_dir = '/data/datasets/yanfu/VOCdevkit/VOC2012/JPEGImages'
extracted_img_dir = '/data/datasets/yanfu/detection_data/imgs'
extracted_xml_dir = '/data/datasets/yanfu/detection_data/annotations'
new_name_prefix = 'VOC07'

for xml_name in os.listdir(xml_dir):
    tree = ET.parse(os.path.join(xml_dir, xml_name))
    annos = []
    for child in tree.getroot():
        if child.tag == 'object':
            class_label = VOC_LABELS.index(child.find('name').text)
            if class_label == 6 or class_label == 14 or class_label == 13:
                copyfile(xml_dir + '/' + xml_name, extracted_xml_dir + '/' + new_name_prefix + '_' + xml_name)
                img_name = xml_name.split('.')[0] + '.jpg'
                copyfile(img_dir + '/' + img_name, extracted_img_dir + '/' + new_name_prefix + '_' + img_name)
