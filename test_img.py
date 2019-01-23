import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from ssd import SSD300
from encoder import DataEncoder
from PIL import Image

import metadata

torch.utils.backcompat.keepdim_warning.enabled = True
torch.utils.backcompat.broadcast_warning.enabled = True

# Load model
net = SSD300()
checkpoint = torch.load('weights/overfit_freezed.pth')
net.load_state_dict(checkpoint['net'])
net.eval()

# Load test image
img = Image.open('data/car1.png')
img1 = img.resize((300, 300))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
img1 = transform(img1)

# Forward
loc, conf = net(Variable(img1[None, :, :, :], volatile=True))
# print(loc)
# print(conf)

# Decode
data_encoder = DataEncoder()
boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

import cv2
import numpy as np

img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
if boxes is not None:
    for box, label, score in zip(boxes, labels, scores):
        box[::2] *= img.shape[1]
        box[1::2] *= img.shape[0]
        box = box.int()
        class_text = metadata.label2text_dict[label[0]]
        prob_text = str(score[0])
        top_left = (box[0], box[1])
        bottom_right = (box[2], box[3])
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(img, class_text + ' ' + 'truck', top_left, cv2.FONT_ITALIC, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("object_detection", img)
cv2.waitKey(-1)
