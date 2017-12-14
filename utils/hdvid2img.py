import cv2

video = '../data/soccer_person_small.mp4'
pic_target_dir = '/home/yf/Pictures/unlabelled'
vidcap = cv2.VideoCapture(video)
vid_name = video.split('/')[-1][:-4]
cnt = 0
success = True
while success:
    success, img = vidcap.read()
    resized_img = cv2.resize(img, (640, 360))
    cv2.imwrite(pic_target_dir + '/' + vid_name + '_frame{}.jpg'.format(cnt), resized_img)
    cnt += 1
