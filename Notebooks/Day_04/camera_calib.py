import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)

fnum = 1

undistorted = False

while True:
    rt,img = cap.read()    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if not undistorted:
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners, ret)
    
    # if we don't have enough pbject points yet, display the old distorted image
    # but if we already have enough, we can calibrate the camera
    if len(imgpoints) >= 10:  
        # calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # get optimal camera matrix:
        h, w = img.shape[:2]
        newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]

        if undistorted is False:
            undistorted = True

        # cv2.putText(img, "calibrated", (40,40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        
        # display the undistorted image instead of the distorted one
        cv2.imshow('calibrated image', dst)

    cv2.imshow('image', img)
    
    key = cv2.waitKey(1) 
    if key == 27:
        break
    elif key & 0xFF == ord('s'):
        cv2.imwrite("chess_" + str(fnum) + ".jpg", img)
        fnum = fnum + 1
    elif key & 0xFF == ord('w'):
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)   
            print "point saved" 

cap.release()
cv2.destroyAllWindows()
