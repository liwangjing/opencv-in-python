import numpy as np
import cv2
from matplotlib import pyplot as plt
import show_image as imgHelper


def show_origin():
    imgHelper.show_image('eiffup', img1);
    imgHelper.show_image('eiffdown', img2);


"""generate Gaussian pyramid for image
   dst = [original image, L5, L4, L3, L2, L1, L0] """
def gener_gaussian_py(src):
    G = src.copy(); # G is original image
    dst = [G];
# gai : 6
    for i in xrange(3): # 1 ~ 6
        G = cv2.pyrDown(G);
        dst.append(G);
    return dst; # dst.size is 7


"""generate laplacian pyramid for image """
def gener_laplacian_py(src): # src is an array
# gai :5
    dst = [src[3]]; # dst[0] =L1
# gai : (5, 0, -1)
    for i in xrange(3, 0, -1): # i is 5, 4, 3, 2, 1
        GE = cv2.pyrUp(src[i]);
        L = cv2.subtract(src[i - 1], GE)
        dst.append(L);
    return dst;


def image_blend():
    # generate gaussian and laplacian pyramids for 2 images
    gpImg1 = gener_gaussian_py(img1);
    gpImg2 = gener_gaussian_py(img2);
    lpImg1 = gener_laplacian_py(gpImg1);
    lpImg2 = gener_laplacian_py(gpImg2);

    # add left and right halves of images in each level
    LS = [];
    for la, lb in zip(lpImg1, lpImg2):
        rows, cols, dpt = la.shape;
        # ls = np.hstack((la[:, 0:cols/2], lb[:, cols/2:]));
        ls = np.vstack((la[0:rows / 2, :], lb[rows / 2:, :]));
        LS.append(ls);
    print "LS size: " + str(len(LS))
    # now reconstruct
    ls_ = LS[0];
#gai : (1, 6)
    for i in xrange(1, 4): # iteration : 1, 2, 3, 4, 5
        ls_ = cv2.pyrUp(ls_);
        ls_ = cv2.add(ls_, LS[i]);

#gai
    cv2.imwrite('laplacian has 3 ele.bmp', ls_)
    imgHelper.show_image('blended image', ls_);


def direct_connect():
    # image with direct connecting each half
    rows, cols, dep = img1.shape;
    # real = np.hstack((img1[:, :cols / 2], img2[:, cols / 2:]));
    real = np.vstack((img1[0:rows / 2, :], img2[rows / 2:, :]));
    imgHelper.show_image('direct connect', real)


def image_blend_msk():
    rows, cols, dpt = img1.shape;
    msk_up, msk_dw = create_msk(rows, cols);


def create_msk(row, col):
    array1 = np.ones((row/2, col, 3));
    array2 = np.zeros((row/2, col, 3));
    msk_up = np.vstack((array1, array2));
    msk_dw = np.vstack((array2, array1));

    imgHelper.show_image('mask_up', msk_up);
    imgHelper.show_image('msk_down', msk_dw);
    return msk_up, msk_dw;

def main():
    show_origin();
    image_blend();
    # direct_connect();
    # image_blend_msk();


img1 = cv2.imread('eiffup.bmp');
img2 = cv2.imread('eiffdown.bmp');

main();