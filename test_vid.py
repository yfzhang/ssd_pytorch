import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from ssd import SSD300
from encoder import DataEncoder
import cv2
import time

import metadata

torch.utils.backcompat.keepdim_warning.enabled = True
torch.utils.backcompat.broadcast_warning.enabled = True

weight_file = 'weights/overfit_freezed.pth'
vid_file = 'data/soccer_person_small.mp4'

# load model
net = SSD300()
checkpoint = torch.load(weight_file)
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

cap = cv2.VideoCapture(vid_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

while True:
    ret, img = cap.read()
    start = time.time()
    img_resized = cv2.resize(img, (300, 300))
    img_transformed = transform(img_resized).cuda()
    # Forward
    loc, conf = net(Variable(img_transformed[None, :, :, :], volatile=True))
    # print(loc)
    # print(conf)

    # Decode
    data_encoder = DataEncoder()
    boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0).cpu(), F.softmax(conf.squeeze(0)).data.cpu())

    # print('boxes {}'.format(boxes))
    # print('labels {}'.format(labels))
    # print('scores {}'.format(scores))
    print('detection took {} s'.format(time.time() - start))

    if boxes is not None:
        print('boxes {}'.format(boxes))
        for box, label, score in zip(boxes, labels, scores):
            box[::2] *= img.shape[1]
            box[1::2] *= img.shape[0]
            box = box.int()
            class_text = metadata.label2text_dict[label[0]]
            prob_text = str(score[0])
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
            cv2.putText(img, class_text + ' ' + prob_text, top_left, cv2.FONT_ITALIC, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("object_detection", img)
    cv2.waitKey(5)
