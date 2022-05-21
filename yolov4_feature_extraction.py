# This file is only an example about the use of YOLOv4 for features extraction, and is not imported in the main source of the application. Please refer to yolo.py
import tensorflow as tf
from yolov4.tf import YOLOv4
import cv2
import numpy as np

yolo = YOLOv4()

yolo.config.parse_names("C:/Users/Hadie/Desktop/Thesis/coco.names")
yolo.config.parse_cfg("C:/Users/Hadie/Desktop/Thesis/yolov4.cfg")

yolo.make_model()
yolo.load_weights("C:/Users/Hadie/Desktop/Thesis/yolov4.weights", weights_type="yolo")


def yolo_load_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # height, width, _ = frame.shape
    frame = yolo.resize_image(frame)
    frame = frame / 255.0
    frame = frame[np.newaxis, ...].astype(np.float32)
    return frame


yolo_new_input = yolo.model.input
yolo_hidden_layer = yolo.model.layers[-1].output

yolo_image_features_extract_model = tf.keras.Model(yolo_new_input, yolo_hidden_layer)

# driver code
img = yolo_load_image("C:/Users/Hadie/Desktop/Thesis/flickr8k/512163695_51a108761d.jpg")
features = yolo_image_features_extract_model(img)
print(features.numpy())
print(features.numpy().shape)
