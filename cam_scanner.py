import imutils
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import array_to_latex

NUM_OF_POINTS = 4


def plot_get_points_steps(img, gray, edged, screenCnt, points):
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", img)
    cv2.imshow("Gray img", gray)
    cv2.imshow("Edged", edged)

    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)

    print("STEP 3: Extract points")
    plt.imshow(img, cmap='gray')
    plt.scatter(points[0], points[1], c='r', s=0.5)
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('with_dots.png')
    plt.show()


def get_points(img, create_plot=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = np.array([])
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    points = np.squeeze(screenCnt).T

    if create_plot:
        plot_get_points_steps(img, gray,edged, screenCnt, points)

    return points




def main(im_path):
    img = cv2.imread(im_path)
    corners = get_points(img)



if __name__ == '__main__':
    im_path = sys.argv[1]
    main(im_path)