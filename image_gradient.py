import numpy as np
import cv2
from matplotlib import pyplot as plt
import show_image as img_helper

img = cv2.imread('flower.jpg', 0);

def high_pass_filter():
    global img;

    laplacian = cv2.Laplacian(img, cv2.CV_64F); # src, ddepth
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5); # preserve the vertical edge
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5); # preserve the horizontal edge

    img_helper.show_image('origin', img);
    img_helper.show_image('laplacian', laplacian);
    img_helper.show_image('sobelX', sobelx);
    img_helper.show_image('sobelY', sobely);

# First argument is our input image.
# Second and third arguments are our minVal and maxVal respectively.
# Third argument is aperture_size.
# It is the size of Sobel kernel used for find image gradients. By default it is 3.
# Last argument is L2gradient which specifies the equation for finding gradient magnitude.
# If it is True, it uses the equation mentioned above which is more accurate,
# otherwise it uses this function: 𝐸𝑑𝑔𝑒_𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡 (𝐺) = |𝐺𝑥| + |𝐺𝑦|. By default, it is False.
def canny_filter():
    img = cv2.imread('flower.jpg', 0);
    img_helper.show_image('flower', img);

    edges = cv2.Canny(img, 100, 200); # (src, threshold1, threshold2)
    img_helper.show_image('canny edges', edges)


def main():
    high_pass_filter();
    canny_filter();

main();