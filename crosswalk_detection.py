import sys
import os

# Crosswalk Detection
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Image Resize Function
def resize_image(image):
    return cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)

# Load Image Function
def load_image_into_numpy_array(path):
    image_np = np.array(Image.open(path))

    if image_np.shape != (640, 640):
        image_np = resize_image(image_np)
        
    return image_np

class CrossWalkStopDetection():
    def __init__(self, input = 0):
        self.input_stream = input
        self.crosswalk_accuracy = 0.6
        self.crosswalk_labels = '../training/workspace/crosswalks/annotations/label_map.pbtxt'
        self.crosswalk_model = '../training/workspace/crosswalks/exported-models/faster_rcnn_resnet101_v1_640x640' + "/saved_model"
        self.vehicle_model = './yolov5n.pt'
        self.strong_sort_weights = './osnet_x0_25_market1501.pt'
        self.save_dir = './output/'

        self.margin_of_error = 10
        self.distance_to_crosswalk = 300
        self.frame_rate = 60
        self.stop_time = 1

        self.postions = {}
        self.crosswalk_mappings = {}
        self.scale_factors = {}
        self.crosswalk_centers = []

        # sample crosswalk centers
        # self.crosswalk_centers = [[1,486,636,688],[1,607,1401,749],[631,475,1359,516],[1320,500,1904,635]]
        # self.crosswalk_centers = [[397, 634, 1072, 678], [1,654,400,818], [287,739,1627,910], [1088,656,1791,766]]

        self.has_stoped = {}


    def first_frame(self):
        vid = cv2.VideoCapture(self.input_stream)
        for i in range(10):
            ret, frame = vid.read()
        vid.release()
        cv2.destroyAllWindows()
        return frame

    def bounding_box_center(self, box):
        return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

    def detect_crosswalks(self):
        frame = self.first_frame()
        category_index = label_map_util.create_category_index_from_labelmap(self.crosswalk_labels, use_display_name=True)
        # Load saved model and build the detection function
        detect_fn = tf.saved_model.load(self.crosswalk_model)
        image_np = frame
        
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections.items()
        }
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)
        
        filter_indices = np.where(detections['detection_scores'] > self.crosswalk_accuracy)
        crosswalks = detections['detection_boxes'][filter_indices]
        self.crosswalk_centers = crosswalks

    def distance_formula(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_crosswalk(self, uid, center, crosswalks):
        closets = []
        min = float('inf')
        min_coordinates=(0,0)

        for crosswalk in crosswalks:
            x_avg = crosswalk[0] + crosswalk[2] // 2
            y_avg = crosswalk[1] + crosswalk[3] // 2
            if self.distance_formula(center, (x_avg, y_avg)) < min:
                
                min = self.distance_formula(center, crosswalk)
                min_coordinates = crosswalk
            closets.append(self.distance_formula(center, crosswalk))
        self.crosswalk_mappings[uid] = min_coordinates

    def handle_detection(self, uid, center, scale_factor):
        self.scale_factors[uid] = scale_factor
        if uid not in self.postions:
            self.closest_crosswalk(uid, center, self.crosswalk_centers)
            dist = self.distance_formula(center, self.crosswalk_mappings[uid])
            self.has_stoped[uid] = False
            self.postions[uid] = [dist]
        elif self.has_stoped[uid] == False:
            dist = self.distance_formula(center, self.crosswalk_mappings[uid])
            self.postions[uid].append(dist)

    def check_stopped(self):
        for uid in self.postions:
            if len(self.postions[uid]) > 4:
                has_stopped = False

                for index, cross_dist in enumerate(self.postions[uid]):
                    if cross_dist <= self.scale_factors[uid]*3.5:
                        time_ago = self.frame_rate*self.stop_time
                        if index-time_ago >= 0:
                            if abs(self.postions[uid][index-time_ago] - cross_dist) <= self.margin_of_error:
                                has_stopped = True

                self.has_stoped[uid] = has_stopped

        print(self.has_stoped)

if __name__ == '__main__':
    crosswalk = CrossWalkStopDetection(input='./20221206_142455.mp4')
    # frame = crosswalk.first_frame()
    # detections = crosswalk.detect_crosswalks(frame)
    detections = [[397, 634, 1072, 678], [1,654,400,818], [287,739,1627,910], [1088,656,1791,766]]
    print(detections)
    # cv2.imshow('First Frame', detections)
    # cv2.waitKey(0)