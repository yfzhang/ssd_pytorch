"""
Generate a train list to overfit
"""

import os
import random
import math

xml_dir = '/data/datasets/yanfu/detection_data/annotations'
img_dir = '/data/datasets/yanfu/detection_data/imgs'

xml_list = os.listdir(xml_dir)
train_xml_list = []
train_list_file = 'overfit_list.txt'

# create train list
for item in xml_list:
    pre_name = item.split('_')[0]
    if pre_name == 'pos' or pre_name == 'gascola':
        train_xml_list.append(item)

train_file = open('overfit_list.txt', 'w')
for xml_name in train_xml_list:
    train_file.write(xml_name)
    train_file.write('\n')
train_file.close()

labels = (
    'person',  # 0
    'car',  # 1
    'motorbike',  # 2
)

# convert annotation
import xml.etree.ElementTree as ET

train_file = open(train_list_file, 'r')
train_list = [line.rstrip() for line in train_file]
f = open('overfit.txt', 'w')
for xml_name in train_list:
    img_name = xml_name[:-4] + '.jpg'
    f.write(img_name + ' ')

    tree = ET.parse(os.path.join(xml_dir, xml_name))
    annos = []
    for child in tree.getroot():
        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            if child.find('name').text in labels:
                class_label = labels.index(child.find('name').text)
                annos.append('%s %s %s %s %s' % (xmin, ymin, xmax, ymax, class_label))
    f.write('%d %s\n' % (len(annos), ' '.join(annos)))
train_file.close()
f.close()
