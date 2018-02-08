import numpy as np
import cv2
from matplotlib import pyplot as plt

print cv2.__version__

video = cv2.VideoCapture("highway.mp4")

while True:
    ret, frame = video.read()

    if frame is None:
        break

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()