"""
Copy annotated images to dataset
"""

import os
import random
from shutil import copyfile

source_dir = '/home/yf/Pictures/tmp'
label_target_dir = '/data/datasets/yanfu/detection_data/annotations'
img_target_dir = '/data/datasets/yanfu/detection_data/imgs'

imgs = os.listdir(source_dir)
for item in imgs:
    if item.split('.')[-1] == 'xml':
        print(item)
        copyfile(source_dir + '/' + item, label_target_dir + '/' + item)
        img = item.split('.')[0] + '.jpg'
        copyfile(source_dir + '/' + img, img_target_dir + '/' + img)
