import cv2
import numpy as np
from matplotlib import pyplot as plt

def pause(key):
    while cv2.waitKey(10) != ord(key):
        continue

cap = cv2.VideoCapture(0)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []

while True:
    ret, frame = cap.read()

    if ret != True:
        continue

   # cv2.imshow("frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if faces == []:
        faces = cascade.detectMultiScale(frame, 1.05, 5)

        max_num, max_area = 0, 0
        for i in range(len(faces)):            
            (x, y, w, h) = faces[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if (w*h) > max_area:
                max_num = i

        (c,r,w,h) = faces[max_num]
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # calculate histogram maximums:
        hue_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        sat_hist = cv2.calcHist([hsv_roi],[1],None,[256],[0,256])
        val_hist = cv2.calcHist([hsv_roi],[2],None,[256],[0,256])

        max_hue = np.argmax(hue_hist)
        max_sat = np.argmax(sat_hist)
        max_val = np.argmax(val_hist)

        mask = cv2.inRange(hsv_roi, (max_hue-40, max_sat-40, max_val-50), 
                                    (max_hue+40, max_sat+40, max_val+50))

        cv2.imshow("mask", mask)
       
        roi_hist = cv2.calcHist([hsv_roi], [0,1], mask, [180,256], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0,1], roi_hist, [0, 180, 0, 256], 1)

        cv2.imshow("backproj", dst)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, (0,255,0), 3)
        cv2.imshow('img2', img2)

    
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        faces = []
    elif key == 27:
        break

cv2.destroyAllWindows()
cap.release()
