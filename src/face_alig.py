from mtcnn.mtcnn import MTCNN
import cv2
import time
import numpy as np


class FaceAlign:
    
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, image):
        """
        Return a list of prediction for face detection
        """

        return self.detector.detect_faces(image)

    def build_box_for_track(self, predictions):
        bounding_boxes = []

        for predicion in predictions:
            rectangle = predicion['box']
            rectangle.append(predicion['confidence'])

            bounding_boxes.append(rectangle)

        return bounding_boxes

    def annotate_image(self, image, predictions):
        
        for detection in predictions:
            rectangle = detection['box']
            points = detection['keypoints']
            prob = detection['confidence']

            #points = np.transpose(points)

            cv2.putText(image, str(prob),
                        (int(rectangle[0]), int(rectangle[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0))

            cv2.rectangle(image,
              (rectangle[0], rectangle[1]),
              (rectangle[0]+rectangle[2], rectangle[1] + rectangle[3]),
              (255, 0, 0), 1)

            cv2.circle(image,(points['left_eye']), 2,  (0, 255, 0), 2)
            cv2.circle(image,(points['right_eye']), 2,  (0, 255, 0), 2)
            cv2.circle(image,(points['nose']), 2,  (0, 255, 0), 2)
            cv2.circle(image,(points['mouth_left']), 2,  (0, 255, 0), 2)
            cv2.circle(image,(points['mouth_right']), 2,  (0, 255, 0), 2)

        return image

    def annodate_image_simple(self, frame, rectangles):
        for rectangle in rectangles:
            cv2.putText(frame, str(rectangle[4]),
                        (int(rectangle[0]), int(rectangle[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 
                        (0, 255, 0),
                        2)
            """
            cv2.rectangle(frame,
              (int(rectangle[0]), int(rectangle[1])),
              (int(rectangle[0]+rectangle[2]), int(rectangle[1] + rectangle[3])),
              (255, 0, 0), 1)
            """

        return frame