import numpy as np
import cv2
from matplotlib import pyplot as plt


def plt_show_image(window1, img1, window2, img2):
    plt.subplot(121), plt.imshow(img1), plt.title(window1);
    plt.subplot(122), plt.imshow(img2), plt.title(window2);
    plt.show();


# pass in the name of the window
def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL);
    cv2.imshow(window_name, img);
    print window_name+str(img.shape)
    # cv2.setMouseCallback(window_name, get_pixel);
    k = cv2.waitKey(0) & 0xFF; # press '0', it will wait infinitely for a key stroke
    if k == ord('q'):
        cv2.destroyAllWindows();