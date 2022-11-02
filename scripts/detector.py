#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import namedtuple


# Constants
IMG_TOPIC = "tb3_0/camera/rgb/image_raw"
MODEL_PATH_WEIGHTS = "/home/alagoo1/catkin_ws/src/pursuit_evasion/dnn/yolov3.weights"
MODEL_PATH_CFG = "/home/alagoo1/catkin_ws/src/pursuit_evasion/dnn/yolov3.cfg"


"""Named tuple representing match """
Match = namedtuple('Match', 'center size confidence')


class Detector:
    """Accepts and detects evader"""
    def __init__(self, min_confidence=0.5):
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
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

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


def main():
    # Create detector
    detector = Detector()

    # Initialize subscriber node
    rospy.init_node('DetectionNode')
    rospy.Subscriber(IMG_TOPIC, Image, detector.detect)
    rospy.spin()


if __name__ == '__main__':
    main()

