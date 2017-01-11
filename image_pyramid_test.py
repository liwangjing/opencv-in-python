import cv2
import numpy as np
import show_image as img_helper


def get_gaussian(image):
    # dst = [I, L0, L1, L2]
    G = image.copy();
    dst = [G];
    for i in xrange(3): # 1 ~ 3
        G = cv2.pyrDown(G);
        name = "gauss "+str(i)+": "
        img_helper.show_image(name, G)
        dst.append(G);
    return dst;


def get_laplac(src):
    # [L2, H2, H1, H0]
    dst = [src[3]];
    for i in xrange(3, 0, -1): # 3, 2, 1
        GE = cv2.pyrUp(src[i]);
        L = cv2.subtract(src[i - 1], GE);
        name = "lapla " + str(i) + ": "
        img_helper.show_image(name, L)
        dst.append(L);
    return dst;


def blend():
    img = cv2.imread('eiffup.bmp');
    img_helper.show_image('origin', img);
    gauss = get_gaussian(img);
    lapla = get_laplac(gauss);

    temp = cv2.add(cv2.pyrUp(gauss[3]), lapla[1]);
    temp = cv2.add(cv2.pyrUp(temp), lapla[2]);
    temp = cv2.add(cv2.pyrUp(temp), lapla[3]);

    cv2.imwrite('laplacian image.bmp', temp);
    img_helper.show_image('blended', temp);


blend();