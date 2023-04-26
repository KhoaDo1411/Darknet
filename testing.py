from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from darknet import *

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="image source")
    parser.add_argument("--video_path", type=str, default="",
                        help="video source")
    parser.add_argument("--weights", default="yolov4-tiny-custom_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny-custom.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./data/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--save_dir", type=str, default='/home/khoado1411/Desktop/yolo/yolov4_darknet/output/test.avi',
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
    
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

def img(args):
    # cap=cv2.VideoCapture('video.mp4')
    # while cap.isOpened():
    #     ret,image = cap.read()
    #     if ret:
    image=cv2.imread(args.input)
    detections, width_ratio, height_ratio = darknet_helper(image, width, height)
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    class_colors[label], 2)
            # cv2.imwrite('/content/drive/MyDrive/CV_Embedded_Intern/darknet/output/result2.jpg',image)
    cv2.imwrite('output/1.jpg',image)
    cv2.waitKey(0)

def video(args):
    assert os.path.exists(args.video_path), \
        'The --video_path is not existed: {}'.format(args.video_path)
    assert args.save_dir.endswith(".avi"), 'The --save_dir should be xxx.avi'
    cap_img = cv2.VideoCapture(args.video_path)
    assert cap_img.isOpened(), "Fail to open video:{}".format(args.video_path)
    fps = cap_img.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_img.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_img.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_img.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter(args.save_dir,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    while cap_img.isOpened():
            ret_img, image = cap_img.read()
            if not ret_img:
                break
            detections, width_ratio, height_ratio = darknet_helper(image, width, height)
            for label, confidence, bbox in detections:
                left, top, right, bottom = bbox2points(bbox)
                left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
                cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
                cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            class_colors[label], 2)
            # cv2.imwrite('/content/drive/MyDrive/CV_Embedded_Intern/darknet/output/result2.jpg',image)
            cap_out.write(image)
    cap_img.release()
    cap_out.release()
def camera(args):
    cap=cv2.VideoCapture(0)
    while True:
        ret,image=cap.read()
        if ret:
            detections, width_ratio, height_ratio = darknet_helper(image, width, height)
            for label, confidence, bbox in detections:
                left, top, right, bottom = bbox2points(bbox)
                left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
                cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
                cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            class_colors[label], 2)
                    # cv2.imwrite('/content/drive/MyDrive/CV_Embedded_Intern/darknet/output/result2.jpg',image)
            cv2.imshow('demo',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    args = parser()
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    video(args)
  
