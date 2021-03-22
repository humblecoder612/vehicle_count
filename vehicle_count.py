import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes
import argparse
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='videos/Night Time Traffic Camera video.mp4', help='source')
parser.add_argument('--x1', type=int, default=800, help='x1')
parser.add_argument('--y1', type=int, default=630, help='y1')
parser.add_argument('--x2', type=int, default=1200, help='x2')
parser.add_argument('--y2', type=int, default=700, help='y2')
opt = parser.parse_args()
x1,y1,x2,y2=opt.x1,opt.y1,opt.x2,opt.y2
count=0
phi = 1
weighted_bifpn = True
model_path = 'd1.h5'
image_size = 640
classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
n_classes = 90
colors = [np.random.randint(0, 256, 3).tolist() for _ in range(n_classes)]
_, model = efficientdet(phi=1,
                            weighted_bifpn=True,
                            num_classes=n_classes,
                            score_threshold=0.5)
model.load_weights(model_path, by_name=True)

video_path = opt.source
cap = cv2.VideoCapture(video_path)

fr=0
while True:
    ret, image = cap.read()
    fr=fr+1
    if not ret:
        break
    img = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]
    h, w = image.shape[:2]
    if y1>h or x1> w or y2>h or x2>w:
        print(h,w)
        print('wrong dim, out of bounds')
        break

    image, scale = preprocess_image(image, image_size=image_size)
        # run network
    start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
    indices = np.where(scores[:] > 0.5)[0]
    boxes = boxes[indices]
    labels = labels[indices]
    for b, lab in zip(boxes, labels):
        xmin, ymin, xmax, ymax = b[0],b[1],b[2],b[3]
        centroid=((xmin+xmax)/2,(ymin+ymax)/2)
        if (centroid[0]>=x1 and centroid[0]<=x2) and (centroid[1]>=y1 and centroid[1]<=y2+10):
            count=count+1
            fr=fr+5
            cap.set(1,fr)
    draw_boxes(img, boxes, scores, labels, colors, classes)
    cv2.line(img, (x1,y1), (x2,y2+10), (0, 255, 0), 9)
    cv2.putText(img, 'count : {}'.format(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    #cv2.waitKey(0)


