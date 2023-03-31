### Example script to read in threshold value for pupil detection from keyboard
# Author: Giuseppe P Gava, 31/03/23

import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from matplotlib.widgets import RangeSlider
import cv2


def image_processing(eye_frame, threshold):
    """Performs operations on the eye frame to isolate the iris

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        threshold (int): Threshold value used to binarize the eye frame

    Returns:
        A frame with a single element representing the iris
    """
    kernel = np.ones((5, 5), np.uint8)
    f_frame = cv2.bilateralFilter(eye_frame, 15, 5, 15)
    # f_frame = cv2.GaussianBlur(eye_frame, (5,5), 0)
    # new_frame = cv2.erode(new_frame, kernel, iterations=3)
    new_frame = cv2.threshold(f_frame, threshold, 255, cv2.THRESH_BINARY)[1]
    new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_OPEN, kernel)
    new_frame = cv2.morphologyEx(new_frame, cv2.MORPH_CLOSE, kernel)

    return new_frame, f_frame


def detect_pupil(eye_frame, threshold):
    """Detects the iris and estimates the position of the iris by
    calculating the centroid.

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
    """
    thr_frame, iris_frame = image_processing(eye_frame, threshold)
    contours, _ = cv2.findContours(thr_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key=cv2.contourArea)
    return iris_frame, thr_frame, contours


if __name__=='__main__':

    with open('frames_temp.dat', 'rb') as f:
        frames_r, frames_l, ts = pickle.load(f)
    frame = frames_r[10]
    img = cv2.bilateralFilter(frame, 15, 5, 15)


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.25)

    im = axs[0].imshow(img, cmap='gray')
    axs[1].hist(img[img<255], bins='auto')
    axs[1].set_title('Histogram of pixel intensities')

    # Create the RangeSlider
    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max())

    # Create the Vertical lines on the histogram
    lower_limit_line = axs[1].axvline(slider.val[0], color='k')
    upper_limit_line = axs[1].axvline(slider.val[1], color='k')


    def update(val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]

        # Update the position of the vertical lines
        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    # read in thr value from keyboard
    thr = None
    while not thr:
        try:
            thr = int(input('Input:'))
        except ValueError:
            print("Not a number")

    # extract pupil contour
    f, n, c = detect_pupil(img, thr)
    # visualise result
    plt.figure(figsize=(8,3))
    plt.subplot(121)
    plt.imshow(f, cmap='gray')
    xc = [t[0][0] for t in c[0]]
    yc = [t[0][1] for t in c[0]]
    plt.plot(xc, yc, 'r')
    plt.subplot(122)
    plt.hist(f[f<255], bins='auto')
    plt.vlines(thr, 0, 100, color='r')
    plt.show()