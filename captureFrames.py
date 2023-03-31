"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import pickle as pickle
import time
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
eye_frames_r = []
eye_frames_l = []
time_stamps = []
time_ref = time.time()

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

    if gaze.pupils_located:

        # check if the right eye has been detected
        eye = gaze.eye_right.frame
        # store frames and time
        eye_frames_r.append(gaze.eye_right.frame)
        eye_frames_l.append(gaze.eye_left.frame)
        time_stamps.append(time.time() - time_ref)

    if cv2.waitKey(1) == 27:# press esc to interrupt the script
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
webcam.release()

# show eye frame
with open('frames_temp.dat', 'wb') as f:
    pickle.dump([eye_frames_r, eye_frames_l, time_stamps], f)

