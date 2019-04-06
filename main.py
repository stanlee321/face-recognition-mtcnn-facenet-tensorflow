from src.face_alig import FaceAlign
from src.tracker.sort import Sort

import time
import cv2

def main():

    face_align = FaceAlign()

    tracker = Sort(use_dlib= False) #create instance of the SORT tracker

    kWinName = 'Face Detection and landmark'
    cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)
    src = '/home/stanlee321/Videos/face/6.mp4'
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        _, frame = cap.read()
        frame = cv2.resize(frame, (320,240))
        predictions = face_align.detect(frame)

        bounding_boxes = face_align.build_box_for_track(predictions)

        print('BOUNDINIG...', bounding_boxes)

        trackers = tracker.update(bounding_boxes, frame)


        frame = face_align.annotate_image(frame, predictions)
        
        frame = face_align.annodate_image_simple(frame, trackers)

        duration = time.time() - start_time
        print(duration)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow(kWinName, frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            
            break   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()