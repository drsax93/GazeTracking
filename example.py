"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import matplotlib.pyplot as plt
import pickle as pickle
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
eye_frames = []

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), 
        (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil),
        (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    try:
        eye = gaze.eye_right.frame
        c = gaze.eye_right.pupil.contours
        xc = [t[0][0] for t in c[0]]
        yc = [t[0][1] for t in c[0]]
        ii = gaze.eye_right.pupil.if2
        eye_frames.append(gaze.eye_right.frame)
        cv2.imshow("eye", gaze.eye_right.frame)
        plt.figure()
        plt.imshow(gaze.eye_right.frame, cmap='gray')
        plt.plot(xc, yc, 'r')
        plt.show()
        plt.pause(0.01)
        plt.close('all')
    except:
        pass

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
webcam.release()

# show eye frame
with open('frames_temp.dat', 'wb') as f:
    pickle.dump(eye_frames, f)

