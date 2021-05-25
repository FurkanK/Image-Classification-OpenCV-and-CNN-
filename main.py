import cv2  
from utils import *
import numpy as np
from imutils.object_detection import non_max_suppression
from collections import Counter
import matplotlib.pyplot as plt
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/home/furkan/Downloads/resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="image"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

                                                             
if not detections:
    check_blank(image)
if detections:
    text_detection(image)


def text_detection(img):

    def model():
            tex_detection_model = { 'name':'EAST',
            'archive':'frozen_east_text_detection.tar.gz',
            'member':'frozen_east_text_detection.pb',
            'sha':'fffabf5ac36f37bddf68e34e84b45f5c4247ed06',
            'filename':'frozen_east_text_detection.pb' }
            Model = tex_detection_model['filename']
            return Model

    pathname = (os.path.join("path", model())) 
    net = cv2.dnn.readNet(pathname)



    def detect_blob():
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(image)
        blob=cv2.drawKeypoints(image   keypoints(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        blob = cv2.resize(blob,(320,320))
        blob = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)
        blob = blob.swapaxes(1, 2).reshape(1, 3, 320, 320)
        return blob

    net.setInput(blob)
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
    (scores, coordinates) = net.forward(layerNames)
    row,column = scores.shape[2],scores.shape[3]
    rectangels = []
    confidence = []

    for y in range(0, row):
        score = scores[0, 0, y]
        up = coordinates[0, 0, y]
        right = coordinates[0, 1, y]
        down = coordinates[0, 2, y]
        left = coordinates[0, 3, y]
        angle = coordinates[0, 4, y]

        for index, num in np.ndenumerate(scores[0][0][0]):
            if num>0.5:
                continue
                x=index[0]
                angle = angle[x]

        cos = math.cos(angle)
        sin = math.sin(angle)
        height = up[x] + down[x]
        weight = right[x] + left[x]
        offset = ([offsetX + cos * right[x] + sin * down[x], offsetY - sin * right[x] + cos* down[x]])
        (offsetX, offsetY) = (x * 4.0, y * 4.0) 
        X_end = offset[0]
        Y_end = offset[1]
        X_start = int(endX – w)
        Y_start = int(endY - h)
        rectangels.append((X_start, Y_start, X_end, Y_end))
        confidence.append(score[x])
        text_region = non_max_suppression(np.array(rectangels), probs=confidences)
         if text_region is None:
             print("Metin içermeyen görüntü")
         else:
             print("Metin içeren görüntü")
                
          return text_region
        

def check_blank(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    list1 = gray.tolist()
    x,y = gray.shape
    total_pixel = x*y
    white_color_counter = 0
    for i in range (0,gray.shape[0]):
        white_color_counter += list1[i].count(255)
        
        blank_filled_ratio = white_color_counter / total_pixel
        if blank_filled_ratio > 0.5:
            print("Dilekçe")
        else:
            print("Fatura")
            

