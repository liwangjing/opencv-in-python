import cv2
import numpy as np
import show_image as imgHelper


def get_mask(row, col):
    array1 = np.ones((row / 2, col, 3));
    array2 = np.zeros((row / 2, col, 3));
    msk_up = np.vstack((array1, array2));
    msk_dw = np.vstack((array2, array1));

    imgHelper.show_image('mask_up', msk_up);
    imgHelper.show_image('msk_down', msk_dw);
    return msk_up, msk_dw;


def blend(image):
    row, col, _ = image.shape;
    msk_up, msk_dw = get_mask(row, col);


def main():
    img = cv2.imread('eiffup.bmp');
    blend(img);


main();
