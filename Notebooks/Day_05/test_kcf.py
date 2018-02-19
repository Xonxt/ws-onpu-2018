import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = []
face = None

tracker = cv2.TrackerKCF_create()

while True:

    ret, frame = cap.read()

    if ret == False:
        break

    if face is None:
        faces = cascade.detectMultiScale(frame, 1.05, 5)

        max_num, max_area = 0, 0

        if len(faces) == 0:
            continue

        for i in range(len(faces)):            
            (x, y, w, h) = faces[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
            if (w*h) > max_area:
                max_num = i
        face = faces[max_num]

        (x,y,w,h) = face
        ok = tracker.init(frame, (x,y,w,h))
    else:
        ok, bbox = tracker.update(frame)
        (x,y,w,h) = bbox
        if ok:
            cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), (255,0,255), 2)

    cv2.imshow("video", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key & 0xFF == ord('q'):
        face = None

cap.release()
cv2.destroyAllWindows()