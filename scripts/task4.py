#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Point, Quaternion
from move_base_msgs.msg import MoveBaseActionResult
import math
from collections import namedtuple
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf.transformations
import time


IMG_TOPIC  = "/tb3_0/camera/rgb/image_raw"
GOAL_TOPIC = "/tb3_0/move_base_simple/goal"
POSE_TOPIC  = "/tb3_0/amcl_pose"

IMG_HEIGHT = 480
IMG_WIDTH = 640

FOV = 70
FOV_MULTIPLIER =  math.tan(math.radians(FOV/2)) / (IMG_WIDTH/2)

DIST_STEP = 0.5


MODEL_PATH_WEIGHTS = "/home/alagoo1/catkin_ws/src/pursuit_evasion/dnn/yolov3.weights"
MODEL_PATH_CFG = "/home/alagoo1/catkin_ws/src/pursuit_evasion/dnn/yolov3.cfg"


"""Named tuple representing match """
Match = namedtuple('Match', 'center size confidence')


class Detector:
    """Accepts and detects evader"""
    def __init__(self, min_confidence=0.1):
        """Loads parameters and initializes CvBridge and neural network.

        Args:
            min_confidence: Minimum confidence required for a match to be accepted. Default value 0.5
        """
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Parameters and flags
        self.min_confidence = min_confidence

        # Load neural network and layer info
        self.net = cv2.dnn.readNetFromDarknet(MODEL_PATH_CFG, MODEL_PATH_WEIGHTS)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_layers = self.net.getLayerNames()
        self.out_layers = [self.out_layers[int(l)] for l in self.net.getUnconnectedOutLayers() - 1]

    def _detect_evader(self, image):
        """Detects evader and returns bounding box.

        Args:
            image: Input image

        Returns:
            Match of bounding box center, size and confidence, if human has been detected. Else, None
        """
        # Pre-process image and extract info
        img_h, img_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=False, crop=False)

        # Pass image to network
        self.net.setInput(blob)
        layer_out = self.net.forward(self.out_layers)

        # Filter matches for humans
        top_match = None
        for output in layer_out:
            for detection in output:
                # Extract most probable class and its confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Extract box dimensions
                (x, y, w, h) = detection[0:4] * [img_w, img_h, img_w, img_h]

                # Select top match
                if class_id == 0 and confidence > self.min_confidence:
                    print(class_id)
                    if top_match:
                        if confidence > top_match.confidence:
                            top_match = Match((int(x), int(y)), (int(w), int(h)), confidence)
                    else:
                        top_match = Match((int(x), int(y)), (int(w), int(h)), confidence)

        return top_match

    def detect(self, data):
        """Callback function to be called on image detection."""
        # Convert image to array and detect evader
        img = self.bridge.imgmsg_to_cv2(data)
        match = self._detect_evader(img)
        return match


class Pursuer:
    """Detects location of evader and moves towards it"""
    def __init__(self):
        """Initialize objects and saves data"""
        # Initialize location and pose
        self.first_frame = True
        self.box = None
        self.yaw = None
        self.position = None
        self.frame_id = "map"

        # Initialize detector
        self.detector = Detector()

        # Initialize node, subscriber and publisher
        rospy.init_node("Evader")
        self.sub_img = rospy.Subscriber(IMG_TOPIC, Image, callback=self.move)
        self.sub_pose = rospy.Subscriber(POSE_TOPIC, PoseWithCovarianceStamped, callback=self._save_pose)
        self.publisher = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=10)

        rospy.spin()
    
    def _save_pose(self, data):
        """Updates robot pose."""
        quaternion = data.pose.pose.orientation
        quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        self.yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
        self.position = data.pose.pose.position

    def move(self, data):
        """Detects evader location and publishes goal to topic"""
        # Extract match
        start = time.time()
        match = self.detector.detect(data)
        print(match)
        if match is not None and self.yaw is not None and self.position is not None:
            # Calculate angle
            x = match.center[0] - IMG_WIDTH/2
            theta = math.atan(-x * FOV_MULTIPLIER) + self.yaw

            # Calculate new location
            x = self.position.x + DIST_STEP * math.cos(theta)
            y = self.position.y + DIST_STEP * math.sin(theta)

            # Convert data types and publish
            position = Point(x, y, self.position.z)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
            
            msg = PoseStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = position
            msg.pose.orientation = Quaternion(*quaternion)
            
            self.publisher.publish(msg)


if __name__ == '__main__':
    pursuer = Pursuer()
