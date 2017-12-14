"""
Randomly sample 90% data to create a train list text file
"""

import os
import random
import math

xml_dir = '/data/datasets/yanfu/detection_data/annotations'
img_dir = '/data/datasets/yanfu/detection_data/imgs'

xml_list = os.listdir(xml_dir)
train_num = math.ceil(len(xml_list) * 0.9)
val_num = len(xml_list) - train_num
train_list = random.sample(xml_list, train_num)
val_list = list(set(xml_list)-set(train_list))

train_file = open('train_list.txt', 'w')
val_file = open('val_list.txt', 'w')
for xml_name in train_list:
    train_file.write(xml_name)
    train_file.write('\n')
train_file.close()

for xml_name in val_list:
    val_file.write(xml_name)
    val_file.write('\n')
val_file.close()
