import numpy as np
import cv2
import show_image as imgHelper


def find_contour(src):
    imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY);
    ret, thresh = cv2.threshold(imgray, 127, 255, 0);
    # mode: cv2.CHAIN_APPROX_SIMPLE = remove the duplicate points.
    # cv2.CHAIN_APPROX_NONE = show all the points on contour.
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    # im, contours, hierarchy = cv2.findContours(thresh, 1, 2);

    image = imgray.copy();
    imgHelper.show_image('image', image)
    cnt = contours[0];
    print "len of cnt: " + str(len(cnt));
    M = cv2.moments(cnt);
    print "M: " + str(M)

    # draw contour point
    image = imgray.copy();
    img = cv2.drawContours(image, cnt, -1, (100, 0, 0), 5);
    imgHelper.show_image('draw_contour', img);

    # find the contour area value
    area = cv2.contourArea(cnt);
    print "area: " + str(area);
    area1 = M['m00'];
    print "area from M: " + str(area1);

    # find contour perimeter:
    perimeter = cv2.arcLength(cnt, True); # True: find the closed perimeter length
    print "perimeter: " + str(perimeter);

    # contour approximation
    epsilon = 0.00001 * cv2.arcLength(cnt, True); # 0.00001
    approx = cv2.approxPolyDP(cnt, epsilon, True);
    print "len of approx: " + str(len(approx))
    temp = imgray.copy();
    img1 = cv2.drawContours(temp, approx, -1, (100, 0, 0), 5);
    imgHelper.show_image('approx', img1);

    # convex hull
    hull = cv2.convexHull(cnt);
    print "len of hull: " + str(len(hull));
    print hull;

    # check the contour is convex or not
    print "check convex: " + str(cv2.isContourConvex(cnt));


def bounding_rectangle():
    img2 = cv2.imread('contour3.bmp');
    imgHelper.show_image('contour3', img2);

    src = img2.copy();
    img2_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY);
    ret, thresh = cv2.threshold(img2_gray, 127, 255, 0);
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    cnt = contours[0];

    # straight bounding rectangle
    img = img2.copy();
    x, y, w, h = cv2.boundingRect(cnt);
    img2_rec = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2);
    imgHelper.show_image('img2 straight rect', img2_rec);

    aspect_ratio = float(w)/h; # get the aspect ratio = width/height. 1.07246376812
    print "straight rect ratio: " + str(aspect_ratio);

    area = cv2.contourArea(cnt);
    extent_ratio = float(area) / (w * h); # get the extent = area of contour/bounding rect area. 0.246132001567
    print "extent ratio: " + str(extent_ratio);

    hull = cv2.convexHull(cnt);
    hull_area = cv2.contourArea(hull);
    solidity_ratio = float(area)/hull_area; # solidity = contour_area/ convex_hull_area
    print "solidity" + str(solidity_ratio);

    equivalent_diameter = np.sqrt(4 * area / np.pi); # equivalent_diameter is the diameter of the circle whose area is same as the contour area.
    print "equivalent_diameter: " + str(equivalent_diameter);

    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt); # orientation is the angle at which obj is directed. MA: major axis, ma: minor axis
    print 'orientation: ' + str(angle);

    # mask and pixel points: find all points that comprises the object
    mask = np.zeros(img2_gray.shape, np.uint8);
    print "mask shape: " + str(mask.shape);
    cv2.drawContours(mask, [cnt], 0, 255, -1);
    pixelpoints = np.transpose(np.nonzero(mask));
    print "pixelpoints: " + str(pixelpoints);
    # pixelpoints = cv2.findNonZero(mask);

    # maximum value, minimum value and their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img2_gray, mask);
    print "max, min, loc: "
    print min_val, max_val, min_loc, max_loc;

    # mean color, mean intensity
    mean_val = cv2.mean(img2.copy(), mask = mask);
    print "mean val:" + str(mean_val);

    # extreme points
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0]);
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0]);
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0]);
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0]);
    print 'leftmost: ' + str(leftmost);
    print 'rightmost: ' + str(rightmost);
    print 'topmost: ' + str(topmost);
    print 'bottommost: ' + str(bottommost);

    # rotated rectangle
    rect = cv2.minAreaRect(cnt);
    box = cv2.boxPoints(rect); # data is float
    print "box: " + "\n" + str(box);
    box = np.int0(box); # turn data into integer.
    print "box np.int0" + "\n" + str(box);
    im = cv2.drawContours(img2_rec, [box], 0, (0, 0, 255), 2);
    imgHelper.show_image('rotated rect', im);

    # minimum enclosing circle:
    (x, y), radium = cv2.minEnclosingCircle(cnt);
    center = (int(x), int(y));
    radium = int(radium);
    im = cv2.circle(im, center, radium, (255, 0, 0), 2);
    imgHelper.show_image('min circle', im);

    # fitting an ellipse
    ellipse = cv2.fitEllipse(cnt);
    im = cv2.ellipse(im, ellipse, (0, 255, 0), 2);
    imgHelper.show_image('ellipse', im);

    # fitting a line
    rows, cols, ch = im.shape;
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01);
    lefty = int((-x * vy / vx) + y);
    righty = int(((cols - x) * vy / vx) + y);
    im = cv2.line(im, (cols - 1, righty), (0, lefty), (0, 255, 255), 2);
    imgHelper.show_image('fit line', im);


def main():
    img = cv2.imread('contour2.bmp');
    imgHelper.show_image("origin", img);
    # find_contour(img);
    bounding_rectangle();


main();
