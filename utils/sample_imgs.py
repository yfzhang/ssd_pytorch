"""
Randomly sample images from a directory
"""

import os
import random
from shutil import copyfile

img_source_dir = '/home/yf/Pictures/tmp'
img_target_dir = '/home/yf/Pictures/unlabelled'
num_imgs = 100

imgs = os.listdir(img_source_dir)
samples = random.sample(imgs, num_imgs)
for img in samples:
    copyfile(img_source_dir + '/' + img, img_target_dir + '/' + img)
