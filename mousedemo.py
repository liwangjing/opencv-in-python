import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1);


def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode;

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True;
        ix, iy = x, y;
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1);
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1);
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False;
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1);
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1);


def main_simple():
    global mode;
    cv2.namedWindow('image');
    cv2.setMouseCallback('image', draw_shape);

    while (1):
        cv2.imshow('image', img);
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode;
        elif k == ord('q'):
            break;

    cv2.destroyAllWindows();


img = np.zeros((512, 512, 3), np.uint8);
drawing = False;
mode = True;  # true: draw circle; false: draw rectangle
ix, iy = -1, -1;
main_simple();
