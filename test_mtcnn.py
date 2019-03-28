# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import time
import cv2
import numpy as np

from src.mtcnn.modelhandler import MTCNNHandler

def main():

    model = MTCNNHandler()
    model.setup()

    kWinName = 'Holistically-Nested_Edge_Detection'
    cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        _, frame = cap.read()

        rectangles, points = model.detect(frame)

        duration = time.time() - start_time

        print(duration)

        print(type(rectangles))
        print(rectangles)
        print(points)

        points = np.transpose(points)

        for rectangle in rectangles:
            cv2.putText(frame, str(rectangle[4]),
                        (int(rectangle[0]), int(rectangle[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0))
            cv2.rectangle(frame, (int(rectangle[0]), int(rectangle[1])),
                            (int(rectangle[2]), int(rectangle[3])),
                            (255, 0, 0), 1)
        for point in points:
            for i in range(0, 10, 2):
                cv2.circle(frame, (int(point[i]), int(
                    point[i + 1])), 2, (0, 255, 0))
        
        cv2.imshow("test", frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            
            break   
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
