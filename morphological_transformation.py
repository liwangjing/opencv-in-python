import numpy as np
import cv2
from matplotlib import pyplot as plt
import show_image as image_helper

# this code works for white foreground, and black background image.
# erosion: all the pixels under the mask are '1', the pixel in the center would be 1
# dilation: under the mask, if at least one pixel is 1, then the center pixel is 1.
# ###

img = cv2.imread('ai2.bmp', 0);
image_helper.show_image('image', img);
kernel = np.ones((5, 5), np.uint8); # np.ones(shape, datatype, order='C');


def inverse_image():
    image = cv2.imread('ai.bmp');
    # get a gray scale image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    # inverse the gray scale image, 0 to 1, 1 to 0.
    img = np.invert(img);
    # save image
    cv2.imwrite('ai2.bmp', img);



# pixel in the original image (either 1 or 0) will be considered 1
# only if all the pixels under the kernel is 1(white),
# otherwise it is eroded (made to zero, or black).
def erode():
    global img;
    global kernel;
    # cv2.erode(src,kernel,dst=None, anchor=None, iteration-=None ... )
    erosion = cv2.erode(img, kernel, iterations=1);
    image_helper.show_image('erode', erosion);


def dilate():
    global img;
    global kernel;
    dilation = cv2.dilate(img, kernel, iterations=1);
    image_helper.show_image('dilation', dilation);


def open_image():
    # opening: erosion , then dilation.
    global img;
    global kernel;
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel); # src, operation, kernel
    image_helper.show_image('opening', opening);


def close_image():
    # closing: dilation, then erosion
    global img, kernel;
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel);
    image_helper.show_image('closing', closing);


# difference between dilation and erosion of a image
def gradient_image():
    global img, kernel;
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel);
    image_helper.show_image('gradient', gradient);


# this is the difference between input image and Opening of the image.
def top_hat_image():
    global img;
    top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((20, 20), np.uint8));
    image_helper.show_image('top hat', top_hat);


# difference between closing image and input image
def black_hat_image():
    global img;
    black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((20, 20), np.uint8));
    image_helper.show_image('black hat', black_hat);


def strucure_element():
    mask1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), np.uint8); # (shape, size), rectangle
    mask2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), np.uint8); # ellipse
    mask3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), np.uint8); # shape is cross.


def main():
    erode();
    dilate();
    open_image();
    close_image();
    gradient_image();
    top_hat_image();
    black_hat_image();


main();
