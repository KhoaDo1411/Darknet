# import darknet functions to perform object detections
from darknet import *
import cv2
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4.cfg", "data/obj.data", "yolov4.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
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

def testing():
    cap=cv2.VideoCapture('video.mp4')
    while cap.isOpened():
        ret,image = cap.read()
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
        cv2.imshow('image',image)
        cv2.waitKey(0)
if __name__ == '__main__':
    testing()