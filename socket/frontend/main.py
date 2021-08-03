from re import S
from flask import Flask, redirect, url_for, render_template, Response
import socket
import time
import numpy as np
import cv2

# ----------------------------------------------------------------
# SOCKETS
HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "192.168.80.100"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))

# ----------------------------------------------------------------
# FLASK

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', socket = send)

@app.route('/message/<msg>')
def send_message(msg):
    send(msg)
    return 'Message Sent'

# ----------------------------------------------------------------
# CAMERA
camera = cv2.VideoCapture(0)

# ----------------------------------------------------------------
# CONFIG YOLOV3

"""
==================     STEP2   ===================
Start of: Loading YOLO v3 network
"""

# Loading COCO class labels from file
with open('yolov3/yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]

# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector with the help of 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet('yolov3/yolo-coco-data/yolov3-tiny.cfg',
                                     'yolov3/yolo-coco-data/yolov3-tiny.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)
# print(colours[0])
"""
End of:
Loading YOLO v3 network
"""

# ----------------------------------------------------------------
# GENERATE FRAME

def gen_frames():  
    # Preparing variables for spatial dimensions of the frames
    h, w = None, None
    detect_mode = False
    start = time.time()
    frame_count = 0
    
    while True:
        # camera = cv2.imread('yolov3/output/output.jpg')

        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if detect_mode:
                # Getting spatial dimensions of the frame we do it only once from the very beginning
                # all other frames have the same dimension

                # if w is None or h is None:
                # Slicing from tuple only first two elements
                h, w = frame.shape[:2]

                """
                Start of:
                Getting blob from current frame
                """

                # Getting blob from current frame
                # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
                # frame after mean subtraction, normalizing, and RB channels swapping
                # Resulted shape has number of frames, number of channels, width and height
                # eg.:
                # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                            swapRB=True, crop=False)

                """
                End of: Getting blob from current frame
                """

                """
                ==================     STEP4   ===================
                Start of: Implementing Forward pass
                """

                # Implementing forward pass with our blob and only through output layers
                # Calculating at the same time, needed time for forward pass
                network.setInput(blob)  # setting blob as input to the network
                start = time.time()
                output_from_network = network.forward(layers_names_output)
                end = time.time()

                # Showing spent time for single current frame
                print('Current frame took {:.5f} seconds'.format(end - start))

                """
                End of:
                Implementing Forward pass
                """

                """
                ==================     STEP5   ===================
                Start of: Getting bounding boxes
                """

                # Preparing lists for detected bounding boxes, obtained confidences and class's number
                bounding_boxes = []
                confidences = []
                classIDs = []

                # Going through all output layers after feed forward pass
                for result in output_from_network:
                    # Going through all detections from current output layer
                    for detected_objects in result:
                        # Getting 80 classes' probabilities for current detected object
                        scores = detected_objects[5:]
                        # Getting index of the class with the maximum value of probability
                        class_current = np.argmax(scores)
                        # Getting value of probability for defined class
                        confidence_current = scores[class_current]
                        
                        # # Every 'detected_objects' numpy array has first 4 numbers with
                        # # bounding box coordinates and rest 80 with probabilities
                        # # for every class
                        # print(detected_objects.shape)  # (85,)

                        # Eliminating weak predictions with minimum probability
                        if confidence_current > probability_minimum:
                            # Scaling bounding box coordinates to the initial frame size
                            # YOLO data format keeps coordinates for center of bounding box
                            # and its current width and height
                            # That is why we can just multiply them elementwise
                            # to the width and height
                            # of the original frame and in this way get coordinates for center
                            # of bounding box, its width and height for original frame
                            box_current = detected_objects[0:4] * np.array([w, h, w, h])

                            # Now, from YOLO data format, we can get top left corner coordinates that are x_min and y_min
                            x_center, y_center, box_width, box_height = box_current
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))

                            # Adding results into prepared lists
                            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            classIDs.append(class_current)

                """
                End of: Getting bounding boxes
                """

                """
                ==================     STEP6   ===================
                Start of: Non-maximum suppression
                """

                # Implementing non-maximum suppression of given bounding boxes
                # With this technique we exclude some of bounding boxes if their
                # corresponding confidences are low or there is another
                # bounding box for this region with higher confidence

                # It is needed to make sure that data type of the boxes is 'int'
                # and data type of the confidences is 'float'
                # https://github.com/opencv/opencv/issues/12789
                results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                        probability_minimum, threshold)

                """
                End of: Non-maximum suppression
                """

                """
                ==================     STEP7   ===================
                Start of: Drawing bounding boxes and labels
                """

                # Checking if there is at least one detected object
                # after non-maximum suppression
                if len(results) > 0:
                    # Going through indexes of results
                    for i in results.flatten():
                        # Getting current bounding box coordinates, its width and height
                        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                        # Preparing colour for current bounding box and converting from numpy array to list
                        colour_box_current = colours[classIDs[i]].tolist()
                        
                        # print(type(colour_box_current))  # <class 'list'>
                        # print(colour_box_current)  # [172 , 10, 127]

                        # Drawing bounding box on the original current frame
                        cv2.rectangle(frame, (x_min, y_min),
                                    (x_min + box_width, y_min + box_height),
                                    colour_box_current, 2)

                        # Preparing text with label and confidence for current bounding box
                        text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                            confidences[i])

                        # Putting text with label and confidence on the original image
                        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
            
            frame_count = frame_count + 1
            end = time.time()
            # print(f'[Timer] {int(end - start)} second')

            if int(end - start) / 30 == 1:
                print('1 seconds have {:.5f} frames'.format( frame_count / 30))
                start = time.time() 
                frame_count = 0

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
