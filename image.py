import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_image_cv2():
    img = cv2.imread('C:\Users\jing\Desktop\me before you.png', 0);
    cv2.namedWindow("image", cv2.WINDOW_NORMAL);
    cv2.imshow("image", img);
    k = cv2.waitKey(0) & 0xFF;
    if k == 27:
        cv2.destroyAllWindows();
    elif k == ord('s'):
        cv2.imwrite('C:\Users\jing\Desktop\me before you.jpg', img);
        cv2.destroyAllWindows();


def show_image_plt():
    img = cv2.imread('C:\Users\jing\Desktop\me before you.png', 0);
    plt.imshow(img, cmap='gray', interpolation='bicubic');
    plt.xticks([]), plt.yticks([])  # to hide tick values on x and y axis
    plt.show();


def draw_on_image():
    # create a black image
    img = np.zeros((512, 512, 3), np.uint8);

    # draw a diagonal blue line with thickness of 5 px, from point (0, 0) to (511, 511).
    img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5);
    # draw a green rectangle from top-left to bottom-right
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3);
    # draw a red circle, center at (447, 63), radius is 63,
    # last param is thickness of the line, -1 is a solid circle, 0 is dotted line.
    img = cv2.circle(img, (447, 64), 63, (0, 0, 255), -1);
    # (img, center, axes(major axis, minor axis), angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
    img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (0, 0, 255), -1);  # this is bottom half ellipse
    # polygon:(image, vertices, isClosed, color)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32);
    pts = pts.reshape((-1, 1, 2));
    img = cv2.polylines(img, [pts], True, (0, 255, 255));
    # add , (image, text_content, position coordinates, font, font_size, color, thickness, line type)
    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA);

    cv2.namedWindow("image", cv2.WINDOW_NORMAL);
    cv2.imshow("image", img);
    cv2.waitKey(0);
    cv2.destroyAllWindows();


img = np.zeros((512, 512, 3), np.uint8);

 # mouse callback function:
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1);


def mouse_draw_circle():
    cv2.namedWindow('image');
    cv2.setMouseCallback('image', draw_circle);

    while (1):
        cv2.imshow('image', img);
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break;
    cv2.destroyAllWindows();


def main():
    img = np.zeros((512, 512, 3), np.uint8); # CREATE A BLACK IMAGE
    # show_image_cv2();
    # show_image_plt();
    # draw_on_image();
    mouse_draw_circle();


main();
