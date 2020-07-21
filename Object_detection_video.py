
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
import os
os.chdir( 'C:\\tensorflow3\\models\\research\\object_detection' )
from readText import read_numberPlate2 


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
# from utils import label_map_util
# from utils import visualization_utils as vis_util

from utils import label_map_util

from utils import visualization_utils as vis_util



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test4.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)


detectedVehicleNumber = ''
isRegisteredUser = False
count = 0

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)


    width  = int(video.get(3))  
    height = int(video.get(4))
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, 
        num_detections],feed_dict={image_tensor: frame_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    # All the results have been drawn on the frame, so it's time to display it.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0.80)
    
    boxes =  np.squeeze(boxes)
    
    #this logic is change wthis is true accuracy 1
    if( int(np.squeeze(scores)[0]) == 1):
        count = count + 1
        if count == 2:
            ymin = boxes[0][0]*height
            xmin = boxes[0][1]*width
            ymax = boxes[0][2]*height
            xmax = boxes[0][3]*width
            cropped = frame[int(ymin):int(ymax-3), int(xmin+10):int(xmax-5)]
            cv2.imwrite("testout.png",cropped)
            cv2.imwrite("output.png",frame)
            # resized = cv2.resize(cropped, (800, 800), interpolation = cv2.INTER_AREA)
            # cv2.imwrite("output2.png",resized)
            break
    #
    cv2.imshow('Number Plate Detector', frame)
    cv2.moveWindow("Number Plate Detector", 20,20);
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
def loadOutputimage():
    image = cv2.imread('output.png')
    cropped = cv2.imread('testout.png')
    (detectedVehicleNumber, isRegisteredUser) = read_numberPlate2(cropped)
    if isRegisteredUser:
        strResult ='User is registered: ' + detectedVehicleNumber
        image = cv2.putText(image,strResult,(50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        strResult ='User is not registered: ' + detectedVehicleNumber
        image = cv2.putText(image,strResult,(50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # x, y, w, h = 100, 100, 200, 100
    # sub_img = image[y:y+h, x:x+w]
    # rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    # image[y:y+h, x:x+w] = rect
    cv2.imshow('output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

loadOutputimage()


