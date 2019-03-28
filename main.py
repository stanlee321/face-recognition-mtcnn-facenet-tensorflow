from src.face_alig import FaceAlign
import time
import cv2

def main():

    face_align = FaceAlign()

    kWinName = 'Face Detection and landmark'
    cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)
    src = '/home/stanlee321/Videos/face/6.mp4'
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        _, frame = cap.read()
        frame = cv2.resize(frame, (320,240))
        predictions = face_align.detect(frame)

        frame = face_align.annotate_image(frame, predictions)
        
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