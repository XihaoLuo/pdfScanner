"""
Xihao Luo
CS 72 - Computer Vision Final Project
"""
# Importing packages
import cv2
import numpy as np
import sys


# Finds the contour points given a canny image
def find_contour(canny):
    canny_copy = canny.copy()
    # mode = 1: takes only the extreme outer contour
    # method = 2: takes only the end points of contour
    img, contours, hierarchy = cv2.findContours(canny_copy, mode=1, method=2)

    # Get all the contour areas
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    # Identify the four contour points
    found_contour = []
    while len(areas) != 0:
        max_index = np.argmax(areas)
        contour = contours[max_index]

        areas.pop(max_index)  # remove the largest area from the list

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter  # allow 1% error from actual perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            found_contour = approx
            break

    return found_contour


# A function that checks whether the contour points are correct on the image
def check_contour(scan, contour_pts):
    img = scan.copy()
    h, w = int(img.shape[0] / 2), int(img.shape[1] / 2)
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)
    for pt in contour_pts:
        x = pt[0][0]
        y = pt[0][1]
        img = cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Image', img)
    while cv2.waitKey(5) < 0:
        pass


# Followed: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# Takes in 4 corner points of a rectangle and figure out its orientation
def order_points(contour_pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = contour_pts.sum(axis=1)
    rect[0] = contour_pts[np.argmin(sums)]  # top left
    rect[3] = contour_pts[np.argmax(sums)]  # bottom right

    diffs = np.diff(contour_pts, axis=1)
    rect[1] = contour_pts[np.argmin(diffs)]  # top right
    rect[2] = contour_pts[np.argmax(diffs)]  # bottom left

    return rect


def main():
    scan = cv2.imread('img2.jpg')  # initial size 4000 x 3000
    img = scan.copy()

    # Resizing a little
    h, w = int(img.shape[0] / 2), int(img.shape[1] / 2)
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)
    img_copy = img.copy()

    # Edge Detection
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur
    canny = cv2.Canny(img, 150, 255)  # Canny Edge Detection - NMS + Hysteresis thresholding

    # Find contour
    found_contour = find_contour(canny)
    if len(found_contour) == 0:
        print("error")
        sys.exit(1)
    found_contour = found_contour.reshape(-1, 2).astype(np.float32)
    found_contour = order_points(found_contour)
    corner_pts = np.array([[0, 0],
                           [w, 0],
                           [0, h],
                           [w, h]], dtype='float32')

    # Transformation
    M = cv2.getPerspectiveTransform(found_contour, corner_pts)
    warped = cv2.warpPerspective(img_copy, M, (w, h))

    # Adaptive Threshold
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    fin_img = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 13, 8)

    cv2.imwrite('output2.jpg', fin_img)
    cv2.imshow('Image', fin_img)
    while cv2.waitKey(5) < 0:
        pass


if __name__ == "__main__":
    main()
