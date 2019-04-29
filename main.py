# !/usr/bin/python3

import cv2
import time
import logging
import argparse
import numpy as np

from src.tracker.sort import Sort
from src.face_alig import FaceAlign

parser = argparse.ArgumentParser(description="Introduce the parameters")
parser.add_argument('-i', '--input',  type = str,  default = '0',   help = 'Introduce the path to the video or source')
parser.add_argument('-d', '--debug',  type = bool, default = False, help = 'Save debug info to log file')
parser.add_argument('-s', '--show',   type = bool, default = False, help = 'Set True if you wish to display images')
parser.add_argument('-p', '--skip',   type = int,  default = 1,     help = 'Set the number of frames between detections')
parser.add_argument('-W', '--width',  type = int,  default = 320,   help = 'Introduce Width')
parser.add_argument('-H', '--height', type = int,  default = 240,   help = 'Introduce Height')
args = parser.parse_args()

my_level = logging.INFO
if args.debug:
    my_level = logging.DEBUG

logging.basicConfig(handlers = [logging.FileHandler('./face_log.txt',mode='w'),
                                logging.StreamHandler()], 
                    format = '%(asctime)s : %(message)s',     # format='%(name)s: %(levelname)s - %(asctime)s : %(message)s'
                    level = my_level)

def main():
    # Configure some parameters
    image_size = (args.width,args.height)
    margin = 40
    face_align = FaceAlign(image_size, margin)
    directoryname = 'images/'

    tracker = Sort()  # create instance of the SORT tracker

    kWinName = 'Face Detection and landmark'
    colours = np.random.rand(32, 3)
    scale_rate = 0.9  # if set it smaller will make input frames smaller
    show_rate = 0.9  # if set it smaller will dispaly smaller frames

    cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)
    try:
        cap = cv2.VideoCapture(args.input)
    except:
        logging.error('Could not load source')
    frame_interval = args.skip  # interval how many frames to make a detection,you need to keep a balance between performance and fluency
    frame_counter = 0
    
    while True:
        start_time = time.time()
        _, frame = cap.read()
        frame = cv2.resize(frame, image_size)
        addtional_attribute_list = []
        final_faces = []

        # If stated we skip frames for the detection to make the process faster
        if frame_counter % frame_interval == 0:
            # Make a detection
            predictions = face_align.detect(frame)
            # If some predictions are found we generate the building box
            if len(predictions) > 0:
                addtional_attribute_list, final_faces, frame = face_align.build_box_for_track(frame, predictions, addtional_attribute_list, final_faces)

        trackers = tracker.update(final_faces, image_size, directoryname, addtional_attribute_list, frame)

        for detected_object in trackers:
            # We get the points of the rectangle and the displacement for the label
            (x0, y0, x1, y1, displacement) = detected_object.astype(np.int32)
            cv2.rectangle(  frame,
                            (x0, y0),
                            (x1, y1),
                            colours[displacement % 32, :] * 255,
                            5)

            cv2.putText(frame,
                        'ID : %d' % (displacement),
                        (x0 - 10, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        colours[displacement % 32, :] * 255, 2)

            if final_faces != []:
                cv2.putText(frame,
                            'DETECTOR',
                            (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (1, 1, 1),
                            2)
        
        #frame = face_align.annotate_image(frame, predictions)
        #frame = face_align.annodate_image_simple(frame, trackers)

        duration = time.time() - start_time
        logging.debug('Last duration: {0:.2f}'.format(duration))
        #frame = cv2.resize(frame, (640, 480))

        frame = cv2.resize(frame, (0, 0), fx = show_rate, fy = show_rate)

        if args.show:
            cv2.imshow(kWinName, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()