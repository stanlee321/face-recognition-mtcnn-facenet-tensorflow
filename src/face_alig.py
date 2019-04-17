from mtcnn.mtcnn import MTCNN           # Import from pip
import cv2
import time
import numpy as np

from .tracker.lib.face_utils import judge_side_face


class FaceAlign:
    
    def __init__(self, image_size, margin):
        self.detector = MTCNN()

        self.img_size = image_size
        self.margin = margin

    def detect(self, image):
        """
        Return a list of prediction for face detection
        """

        return self.detector.detect_faces(image)

    def build_box_for_track(self, frame, predictions, addtional_attribute_list, final_faces):
        
        faces = predictions.copy()
        
        face_sums = len(faces)

        final_faces = None

        if face_sums > 0:
            face_list = []
            for i, item in enumerate(faces):
                f = round(item['confidence'], 6)
                if f > 0.99:
                    det = np.squeeze(item['box'])

                    # face rectangle
                    det[0] = det[0] - self.margin
                    det[1] = det[1] - self.margin
                    det[2] = det[0] + det[2] + self.margin
                    det[3] = det[1] + det[3] + self.margin
                                            
                    face_list.append([det[0], det[1], det[2], det[3], f ])

                    # face cropped
                    bb = np.array(det, dtype=np.int32)
                    frame_copy = frame.copy()
                    cropped = frame_copy[bb[1]:bb[3], bb[0]:bb[2], :]

                    # use 5 face landmarks  to judge the face is front or side
                    facial_landmarks = []

                    for k, v  in item['keypoints'].items():
                        facial_landmarks.append(v)
                    """
                    for (x, y) in facial_landmarks:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    cv2.rectangle(frame,
                                    (det[0], det[1]),
                                    (det[0] + det[2], det[1] + det[3]),
                                    (255, 0, 0), 1)
                    """
                    dist_rate, high_ratio_variance, width_rate = judge_side_face( np.array(facial_landmarks))

                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                    item_list = [cropped, item['confidence'], dist_rate, high_ratio_variance, width_rate]
                    
                    addtional_attribute_list.append(item_list)

            final_faces = np.array(face_list)

        return addtional_attribute_list, final_faces, frame

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