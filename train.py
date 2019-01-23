import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from ssd import SSD300
from datagen import ListDataset
from multibox_loss import MultiBoxLoss
from torch.autograd import Variable

from logger import Logger

# parameters
start_epoch = 0
end_epoch = 200
batch_size = 32
lr = 1e-3
resume = False
torch.utils.backcompat.keepdim_warning.enabled = True
torch.utils.backcompat.broadcast_warning.enabled = True
train_list = './utils/train.txt'
val_list = './utils/test.txt'
ckpt_name = 'coco'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
train_set = ListDataset(list_file=train_list, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
valid_set = ListDataset(list_file=val_list, train=False, transform=transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

net = SSD300()
if resume:
    print('resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    net.load_state_dict(torch.load('./weights/ssd_initializedVGG.pth', map_location=lambda storage, loc: storage))

torch.cuda.set_device(0)
# net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
net.cuda()
cudnn.benchmark = True

criterion = MultiBoxLoss()
optimizer = optim.SGD(net.gen_trainable_params(), lr=lr, momentum=0.9, weight_decay=1e-4)
best_loss = float('inf')
step = 0
logger = Logger('./logs')


def train(epoch):
    print('*** epoch: {} ***'.format(epoch))
    start = time.clock()
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(tqdm(train_loader)):
        images = images.cuda()
        loc_targets = loc_targets.cuda()
        conf_targets = conf_targets.cuda()

        images = Variable(images)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]

        print('train: loss {:.3f}'.format(train_loss / (batch_idx + 1)))
        info = {'train_loss': train_loss / (batch_idx + 1), 'lr': optimizer.param_groups[0]['lr']}
        global step
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)
        step += 1
    print('*** epoch ends, took {}s ***'.format(time.clock() - start))


def test(epoch):
    net.eval()
    valid_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(tqdm(valid_loader)):
        images = images.cuda()
        loc_targets = loc_targets.cuda()
        conf_targets = conf_targets.cuda()

        images = Variable(images, volatile=True)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        valid_loss += loss.data[0]
        print('test: loss {:.3f}'.format(valid_loss / (batch_idx + 1)))
        info = {'valid_loss': valid_loss / (batch_idx + 1)}
        global step
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

    # Save checkpoint.
    global best_loss
    valid_loss /= len(valid_loader)
    if valid_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + ckpt_name + '_epoch' + str(epoch) + '.pth')
        best_loss = valid_loss


for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
