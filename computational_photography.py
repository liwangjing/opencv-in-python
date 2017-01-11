import numpy as np
import cv2
import show_image as imgHelper


def create_noise(src):
    noise = np.random.randn(*src.shape)*10
    return noise


def add_noise(src, noise):
    noisy = src + noise;
    return noisy


def noisy_img(src):
    noise = create_noise(src);
    noisy = add_noise(src, noise);
    noisy = np.uint8(np.clip(noisy, 0, 255));
    return noisy


def denoise_gray_image(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    imgHelper.show_image('gray', im_gray);
    gray = np.float64(im_gray);

    noisy = noisy_img(gray);
    imgHelper.show_image('noisy', noisy)

    dst = cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21);
    imgHelper.show_image('denoised', dst)


def denoise_color_image(img):
    temp = np.float64(img);
    noisy = noisy_img(temp);
    imgHelper.show_image('noisy color', noisy)

    dst = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21);
    imgHelper.show_image('denoise color', dst)

def denoise(img):
    denoise_gray_image(img);
    denoise_color_image(img);


def inpaint(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    msk = create_mask(gray);

    dst_ns = cv2.inpaint(img, msk, 3, cv2.INPAINT_NS);
    imgHelper.show_image('NS inpaint', dst_ns);

    dst_telea = cv2.inpaint(img, msk, 3, cv2.INPAINT_TELEA);
    imgHelper.show_image('TELEA inpaint', dst_telea);


def create_mask(img):
    ret, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY_INV);
    imgHelper.show_image('mask', thresh);
    return thresh;

def main():
    img = cv2.imread('lena.png');
    imgHelper.show_image('origin', img);

    img2 = cv2.imread('lena_damaged.bmp');
    imgHelper.show_image('impaint origin', img2);

    # denoise(img);
    inpaint(img2);


main();

