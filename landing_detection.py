'''
Tests the landing detection and relative orientation algorithm used for precision landings
'''

import numpy as np
import tensorflow as tf
import cv2
from threading import Thread
from simple_cb import simplest_cb


class LandingDetection:

    # src is the mount point id of the camera
    # graph_location is the location of the landing zone detection model
    def __init__(self, src, graph_location):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(3, 400)  # width
        self.cap.set(4, 400)  # height
        self.cap.set(5, 30)  # fps

        # Set default graph in tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.GFile(graph_location, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

        self.initialized = False
        self.kill = False

    # Returns 0 if the landing zone is centered along one dimension
    # Returns -1 if too high or too left
    # Returns 1 if too low or too right
    def compare(self, center):
        # 5% buffer
        if center < 0.45:
            return -1
        elif center > 0.55:
            return 1
        else:
            return 0

    # Returns the location of the landing zone relative to the center of the image
    # Assumes that the aircraft's downward facing camera is centered, but this can be quickly adapted
    # to add a constant offset in both the x and y dimensions
    # eg. [0,0] is centered, [-1,1] is low left, [1,-1] is high right
    def locate(self, box):
        y_center = (box[2] + box[0]) / 2  # location of bounding box center as a fraction of the total image
        x_center = (box[3] + box[1]) / 2

        return [self.compare(x_center), self.compare(y_center)]

    # Returns 0 if the zoom level is in the desired range
    # Returns 1 if zoomed out too much
    # returns -1 if zoomed in too much
    def zoom(self, box):
        y_size = box[2] - box[0]  # width of the bounding box as a fraction of the total image
        x_size = box[3] - box[1]

        # While the number of pixels should be the same since the landing zone is square,
        # angle relative to the plane can skew the image
        focus_dimension = max(x_size, y_size)

        # Fraction of screen consumed by the bounding box
        if focus_dimension < 0.6:
            return 1
        elif focus_dimension > 0.8:
            return -1
        else:
            return 0

    # Returns a mask that only includes colors in a desired range
    # hsv is an image with a hsv colorspace
    def filter(self, hsv, low, high, sat=100, val=100):
        mask = cv2.inRange(hsv, (low,sat,val), (high,255,255))
        # Accounts for small anomalies in the image caused by artifacts on the lens, sensor, ground, etc.
        return cv2.GaussianBlur(mask,(5,5),5)

    # Returns the center of the largest blob of color in the prefiltered mask
    def locate_color(self, mask):
        thresh = cv2.threshold(mask, 40, 255, 0)[1]
        # Finds all the blobs
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)  # largest blob
            x,y,w,h = cv2.boundingRect(c)
            center_x = (w/2) + x
            center_y = (h/2) + y
            return [center_x, center_y]
        else:
            return None

    # Assigns a location code based on where the color is relative to another
    # Relative codes in a single dimension are defined in classify_dimension()
    # eg. 2 means the color is lower and to the left
    def classify_color(self, hDim, wDim):
        classification_matrix = [[1,2,3],[4,5,6],[7,8,9]]
        return classification_matrix[hDim][wDim]

    # Assigns a relative position based on the distance from one color center to another along a single dimension
    # Used for both dimensions
    def classify_dimension(self, dif, buf):
        if abs(dif) < buf:      # Considered to be on the same line
            return 2
        elif dif >= 0:          # Considered to be to the left or higher
            return 1
        else:                   # Considered to be to the right or lower
            return 0

    # Returns where the blobs of red, blue, and yellow are located relative to the green blob
    # Only accurate when the green blob is located relatively close to an axis of the cartesian plane centered at the
    # center of the bounding box. This is intentional
    def relative_orientation(self, red, blue, yellow, green, height, width):
        # buffers account for errors in the calculation of the blob location without sacrificing any accuracy
        wBuffer = width * 0.2
        hBuffer = height * 0.2
        orientation_code = [None, None, None]
        for index, color in enumerate([red, yellow, blue]):
            wDif = green[0] - color[0]
            hDif = green[1] - color[1]

            wDim = self.classify_dimension(wDif, wBuffer)
            hDim = self.classify_dimension(hDif, hBuffer)

            orientation_code[index] = self.classify_color(hDim, wDim)
        
        return orientation_code

    # Returns a position code based on the orientation of the blobs
    # Can be combined with orientation_target()
    def validate_orientation(self, orientation):
        valid_orientations = [
            [3,8,2],
            [2,5,8],
            [8,6,5],
            [5,4,6],
            [6,7,4],
            [4,1,7],
            [7,3,1],
            [1,2,3]
        ]

        return valid_orientations.index(orientation) if orientation in valid_orientations else None

    # Returns the rotation necessary for proper landing orientation based on the code returned from validate_orientation
    def orientation_target(self, id):
        target_dict = {
            0: 0,
            1: 45,
            2: 90,
            3: 135,
            4: 180,
            5: -135,
            6: -90,
            7: -45
        }

        return target_dict.get(id)

    # Main detection loop that grabs a camera frame each iteration and processes it
    # First, the image is passed through a white balance to account for lighting variances
    # Second, the image is passed through the TensorFlow object detection model that returns a bounding box
    def detect(self):
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            with tf.Session(config=sess_config) as sess:
                tensor_dict = {}
                tensor_dict['detection_boxes'] = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
                tensor_dict['detection_scores'] = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                self.initialized = True

                while True:
                    ret, image = self.cap.read()  # grab image
                    image = simplest_cb(image,1)
                    image_expanded = np.expand_dims(image, axis=0)  # match image to model's expected input

                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_expanded})
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    # take the bounding box with the highest confidence if that confidence is higher than 90%
                    if(output_dict['detection_scores'][0] > .9):
                        height, width = image.shape[:2]

                        box = output_dict['detection_boxes'][0]

                        relative_location = self.locate(box)  # find the center of the bounding box relative to the original image

                        if relative_location == [0,0]:
                            relative_zoom = self.zoom(box)
                            cv2.putText(image, "Zoom: {}".format(str(relative_zoom)),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(image, "Location X: {}".format(str(relative_location[0])),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(image, "Location Y: {}".format(str(relative_location[1])),(10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

                        # Get corners of the bounding box in terms of pixels
                        y_min = int(box[0]*height)
                        y_max = int(box[2]*height)
                        x_min = int(box[1]*width)
                        x_max = int(box[3]*width)

                        # Crop out the bounding box and get its dimensions
                        croppedImage = image[y_min:y_max, x_min:x_max]
                        croppedHeight, croppedWidth = croppedImage.shape[:2]

                        # Filter Colors
                        croppedHSV = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)  # convert colorspace for easier isolation
                        maskRed = self.filter(croppedHSV,160,175,80,110)
                        maskBlue = self.filter(croppedHSV,90,100,170,120)
                        maskYellow = self.filter(croppedHSV,20,35,60,120)
                        maskGreen = self.filter(croppedHSV,75,85,120)

                        # Find Contours (Blobs)
                        contourRed = self.locate_color(maskRed)
                        contourBlue = self.locate_color(maskBlue)
                        contourYellow = self.locate_color(maskYellow)
                        contourGreen = self.locate_color(maskGreen)

                        if None not in [contourRed, contourBlue, contourYellow, contourGreen]:
                            cv2.circle(image,(int(contourRed[0]) + x_min,int(contourRed[1]) + y_min), 5, (0,0,255), -1)
                            cv2.circle(image,(int(contourBlue[0]) + x_min,int(contourBlue[1]) + y_min), 5, (255,0,0), -1)
                            cv2.circle(image,(int(contourYellow[0]) + x_min,int(contourYellow[1]) + y_min), 5, (255,255,0), -1)
                            cv2.circle(image,(int(contourGreen[0]) + x_min,int(contourGreen[1]) + y_min), 5, (0,255,0), -1)

                            # Must be in proper zoom before changing orientation
                            orientation = self.relative_orientation(contourRed, contourBlue, contourYellow, contourGreen, croppedHeight, croppedWidth)
                            orientation_id = self.validate_orientation(orientation)
                            orientation_delta = self.orientation_target(orientation_id)

                            cv2.putText(image, "Orientation: {}".format(str(orientation_id)),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(image, "Delta: {}".format(str(orientation_delta)),(10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

                        # Find dead center of the landing pad
                        # This is used when we are zoomed in too much to use the colored blobs for guidance
                        maskX = cv2.inRange(croppedHSV, (0,0,210), (180,50,255))
                        maskX = cv2.GaussianBlur(maskX,(5,5),5)

                        thresh = cv2.threshold(maskX, 40, 255, 0)[1]
                        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if len(contours) != 0:
                            c = max(contours, key=cv2.contourArea)
                            x,y,w,h = cv2.boundingRect(c)
                            center_x = (w/2) + x
                            center_y = (h/2) + y
                            cv2.circle(image,(int(center_x) + x_min,int(center_y) + y_min), 5, (0,0,255), -1)

                        # Highlight the bounding box
                        top_left = (x_min,y_min)
                        bot_right = (x_max,y_max)
                        cv2.rectangle(image,top_left,bot_right,(0,0,0),3)

                    else:
                        # The model did not find anything
                        print("Nothing Found")

                    # Blow up image 4:1 before displaying it on screen
                    image = cv2.resize(image, (1600, 1600))
                    cv2.imshow("Landing Detection", image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        break

                    if(self.kill):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        break

    # Starts the detection loop in the background
    def detect_start(self):
        Thread(target=self.detect, args=()).start()
        print("Landing_Detection Started")
