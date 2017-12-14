"""
Convert VOC PASCAL 2007/2012 xml annotations to a list file.
"""

import os
import xml.etree.ElementTree as ET

labels = (
    'person',  # 0
    'car',  # 1
    'motorbike',  # 2
)

xml_dir = '/data/datasets/yanfu/detection_data/annotations'
train_list_file = 'train_list.txt'
val_list_file = 'val_list.txt'

train_file = open(train_list_file, 'r')
train_list = [line.rstrip() for line in train_file]
f = open('train.txt', 'w')
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

val_file = open(val_list_file, 'r')
val_list = [line.rstrip() for line in val_file]
f = open('val.txt', 'w')
for xml_name in val_list:
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