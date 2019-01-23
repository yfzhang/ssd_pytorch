"""
Generate image list for training
"""

import os

xml_dir = '/data/datasets/yanfu/detection_data/annotations'
img_dir = '/data/datasets/yanfu/detection_data/imgs'

xml_list = os.listdir(xml_dir)
train_xml_list = []
train_list_file = 'train_list.txt'
train_file = 'train.txt'
test_list_file = 'test_list.txt'
test_file = 'test.txt'

# create train list
for item in xml_list:
    train_xml_list.append(item)

f = open(train_list_file, 'w')
for xml_name in train_xml_list:
    f.write(xml_name.split('.')[0])
    f.write('\n')
f.close()

labels = (
    'person',  # 0
    'car',  # 1
    'motorbike',  # 2
)


# convert annotation
def parse_annotation(input_f, output_f):
    import xml.etree.ElementTree as ET
    import metadata
    f = open(input_f, 'r')
    input_list = [line.rstrip() for line in f]
    f.close()

    f = open(output_f, 'w')
    for name in input_list:
        img_name = name + '.jpg'
        f.write(img_name + ' ')
        tree = ET.parse(os.path.join(xml_dir, name + '.xml'))
        annos = []
        for child in tree.getroot():
            if child.tag == 'object':
                bbox = child.find('bndbox')
                xmin = bbox.find('xmin').text
                ymin = bbox.find('ymin').text
                xmax = bbox.find('xmax').text
                ymax = bbox.find('ymax').text
                if child.find('name').text in metadata.text2label_dict.keys():
                    label = metadata.text2label_dict[child.find('name').text]
                    annos.append('%s %s %s %s %s' % (xmin, ymin, xmax, ymax, label))
        f.write('%d %s\n' % (len(annos), ' '.join(annos)))
    f.close()


parse_annotation(train_list_file, train_file)
parse_annotation(test_list_file, test_file)
