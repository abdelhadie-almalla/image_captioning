from yolov4.tf import YOLOv4
import cv2
import time
from quicksort import quickSort
import numpy as np
import tensorflow as tf

yolo = YOLOv4()
# yolo = YOLOv4(tiny=True)

yolo.config.parse_names("coco.names")
yolo.config.parse_cfg("yolov4.cfg")
# yolo.input_size = (480,640)

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

# yolo.inference(media_path="C:/Users/Hadie/Desktop/yolo/NYC_14th_Street_looking_west_12_2005.jpg")


# the output is sorted according to the area by confidence
def image_path_to_yolo_bounding_boxes(image_path, coco_dict, word_index):
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = yolo.predict(frame, prob_thresh=0.25)
    bboxes = bboxes.tolist()
    n = len(bboxes)
    # for each bounding box, append (area * confidence)
    for i in range(n):
        # bboxes[i].append(bboxes[i][2] * bboxes[i][3] * bboxes[i][5])
        obj_class_name = coco_dict[int(bboxes[i][4])].replace(" ", "")
        if obj_class_name in word_index:
            bboxes[i][4] = word_index[coco_dict[int(bboxes[i][4])].replace(" ", "")]
        else:
            bboxes[i][4] = word_index['<pad>']
    # quickSort(bboxes, 0, n - 1)
    bboxes = np.array(bboxes)
    return bboxes


# raw feature extraction - not bounding boxes
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
# image_path = "C:/Users/Hadie/Desktop/yolo/NYC_14th_Street_looking_west_12_2005.jpg"
# bboxes = image_path_to_yolo_bounding_boxes(image_path)
# print(bboxes)
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image = yolo.draw_bboxes(image, bboxes)
# cv2.imshow("result", image)
# cv2.waitKey()
