from mtcnn.mtcnn import MTCNN
import cv2
import time
import numpy as np
def main():
    detector = MTCNN()

    kWinName = 'Face Detection and landmark'
    cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)
    src = '/home/stanlee321/Videos/face/6.mp4'
    cap = cv2.VideoCapture(src)

    while True:
        start_time = time.time()
        _, frame = cap.read()
        frame = cv2.resize(frame, (320,240))
        predictions = detector.detect_faces(frame)

        for detection in predictions:


            rectangle = detection['box']
            points = detection['keypoints']
            prob = detection['confidence']

            #points = np.transpose(points)

            cv2.putText(frame, str(prob),
                        (int(rectangle[0]), int(rectangle[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0))

            cv2.rectangle(frame,
              (rectangle[0], rectangle[1]),
              (rectangle[0]+rectangle[2], rectangle[1] + rectangle[3]),
              (255, 0, 0), 1)

            cv2.circle(frame,(points['left_eye']), 2,  (0, 255, 0), 2)
            cv2.circle(frame,(points['right_eye']), 2,  (0, 255, 0), 2)
            cv2.circle(frame,(points['nose']), 2,  (0, 255, 0), 2)
            cv2.circle(frame,(points['mouth_left']), 2,  (0, 255, 0), 2)
            cv2.circle(frame,(points['mouth_right']), 2,  (0, 255, 0), 2)
        
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