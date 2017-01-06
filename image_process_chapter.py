import numpy as np
import cv2
from matplotlib import pyplot as plt


# pass in the name of the window
def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL);
    cv2.imshow(window_name, img);
    # cv2.setMouseCallback(window_name, get_pixel);
    k = cv2.waitKey(0) & 0xFF; # press '0', it will wait infinitely for a key stroke
    if k == ord('q'):
        cv2.destroyAllWindows();


def track_with_hsv() :
    img = cv2.imread('opencv_logo.png', 1);
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50]); # corresponding to BGR[50, 43, 40]
    # lower_blue = np.array([110, 255, 255]); # corresponding to BGR[255, 85, 0]
    upper_blue = np.array([130, 255, 255]);

    # threshold the hsv image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue);

    # bitwise-and mask and original image
    res = cv2.bitwise_and(img, img, mask=mask);

    show_image('original', img);
    show_image('mask', mask);
    show_image('res', res);


# find the hsv value for specific color
def get_hsv() :
    blue = np.uint8([[[255, 0, 0]]]);
    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV);
    print hsv_blue;


def threshold_exp():
    img = cv2.imread('lena.png', 0);
    img = cv2.medianBlur(img, 5);

    ret, img1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY);
    img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    show_image('original', img)
    show_image('img1', img1)
    show_image('img2', img2)
    show_image('img3', img3)

# But consider a bimodal image(has two peeks in frequency)
# (In simple words, bimodal image is an image whose histogram has two peaks).
#  For that image, we can approximately take a value in the middle of those
# peaks as threshold value, right ? That is what Otsu binarization does.
# So in simple words, it automatically calculates
# a threshold value from image histogram for a bimodal image.
# (For images which are not bimodal, binarization wont be accurate.)
# For this, our cv2.threshold() function is used, but pass an extra flag,
# cv2.THRESH_OTSU. For threshold value, simply pass zero.
# Then the algorithm finds the optimal threshold value and returns you as the second output,
#  retVal. If Otsu thresholding is not used, retVal is same as the threshold value you used.

def otsu_threshold_exp():
    img = cv2.imread('pepper_noisy.bmp', 0)

    # global threshold : 127
    ret1, th1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU);
    print 'Otsu thresholding: ' + str(ret2)

    # Otsu's threshold after Gaussian filtering:
    # use gaussian filter to filter the image, then use OTSU threshold
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print 'Otsu Gaussian thresholding: ' + str(ret3)

    #  plot all images
    images = [img, 0, th1, img, 0, th2, blur, 0, th3]
    titles = ['original', 'histogram', 'global thresholding (v=127)',
              'original', 'histogram', 'Otsus thresholding',
              'Gaussian filtered image', 'histogram', 'otsus thresholding']

    for i in xrange(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        # image.ravel() turn the matrix into a row vector, row goes first,
        # it returns a reference, change image.ravel().[1] will change the value in the image.
        # use image.flatten(), difference is it returns a copy, it won't change the image.
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256); # num of bins.
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])

        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

    plt.show();


def resize_image():
    # zoom the image up to 2 times
    img = cv2.imread('lena.png')
    print 'img size:' + str(img.shape)
    show_image('origin', img);

    res1 = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    print 'res1 size:' + str(res1.shape)
    show_image('res1', res1);

    height, width = img.shape[:2];
    res2 = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC);
    print 'res2 size:' + str(res2.shape)
    show_image('res2', res2)


def translation_image():
    img = cv2.imread('lena.png', 0)
    rows, cols = img.shape

    M = np.float32([[1, 0, 100], [0, 1, 50]]); # translation matrix, move to right 100, move down 50
    # warpAffine(src, translation matrix, (width, height))
    dst = cv2.warpAffine(img, M, (cols, rows));

    show_image('translated res', dst)


def rotate_image():
    img = cv2.imread('lena.png', 0)
    rows, cols = img.shape;

    # cv2.getRotationMatrix2D(center, angle, scale)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1) # rotate 90 degree inverse-clock-wise
    dst = cv2.warpAffine(img, M, (cols, rows));
    show_image('rotated image', dst)


def affine_image():
    img = cv2.imread('opencv_logo.png', 1);
    show_image('original', img);
    rows, cols, ch = img.shape;

    pts1 = np.float32([[50, 50],[200, 50],[50, 200]]);
    pts2 = np.float32([[10, 100],[200, 50],[100, 250]]);

    M = cv2.getAffineTransform(pts1, pts2);

    dst = cv2.warpAffine(img, M, (cols, rows));
    show_image('result', dst);


# by define an area in the image, turn the curve to straight line
def perspective_transform():
    img = cv2.imread('sudoku');
    rows, cols, ch = img.shape;

    # get 4 points from the img, 3 of them can't be collinear, pass the 4 coordinates
    pts1 = np.float32([[56, 56], [368, 52], [28, 387], [389, 390]]);
    # the corresponding coordinates for the 4 points in pts1.
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]]);

    # pass the 2 coordinate sets to transformation M.
    M = cv2.getPerspectiveTransform(pts1, pts2);

    dst = cv2.warpPerspective(img, M, (300, 300));

    plt_show_image('input', img, 'output', dst);


# Image blurring is achieved by convolving the image with a low-pass filter kernel.
# It is useful for removing noise.
# It actually removes high frequency content (e.g: noise, edges)
# from the image resulting in edges being blurred when this filter is applied.
# (Well, there are blurring techniques which do not blur edges).
# OpenCV provides mainly four types of blurring techniques.
def smooth_image():
    img = cv2.imread('lena.png');
    # np.ones(shape, datatype, order='C')

    # generate a low-pass filter, it will blur the image, remove the noise
    kernel = np.ones((5, 5), np.float32)/25;

    # apply the LPF to the image.
    dst = cv2.filter2D(img, -1, kernel); # -1 is ddepth

    plt_show_image('origin', img, 'result', dst);


def plt_show_image(window1, img1, window2, img2):
    plt.subplot(121), plt.imshow(img1), plt.title(window1);
    plt.subplot(122), plt.imshow(img2), plt.title(window2);
    plt.show();


def blur_image():
    img = cv2.imread('pepper_noisy.bmp');
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

    blur = cv2.blur(img, (5, 5));
    plt_show_image('origin', img, 'blurred', blur);


def gaussian_image():
    img = cv2.imread('pepper_noisy.bmp');
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

    blur = cv2.GaussianBlur(img, (5, 5), 0); # (src, kernel_size, sigmax)
    # show_image('gaussian bluured', blur);
    plt_show_image('origin', img, 'gaussian blurred', blur);


def median_filter_image():
    img = cv2.imread('pepper_noisy.bmp');
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

    blur = cv2.medianBlur(img, 5);  # (src, kernel_size)
    # show_image('median bluured', blur);
    plt_show_image('origin', img, 'median blurred', blur);


def bilateral_filter_image():
    # this blur method preserve the edge
    img = cv2.imread('pepper_noisy.bmp');
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

    blur = cv2.bilateralFilter(img, 9, 75, 75);
    # show_image('bilateral bluured', blur);
    plt_show_image('origin', img, 'median blurred', blur);


def main():
    # get_hsv(); # how to convert bgr to hsv
    # track_with_hsv(); # track an obj in an image with hsv
    # threshold_exp();
    # otsu_threshold_exp();
    # resize_image();
    # translation_image();
    # rotate_image();
    # affine_image();
    # perspective_transform();
    # smooth_image();
    # blur_image();
    gaussian_image();
    median_filter_image();
    bilateral_filter_image();


main();