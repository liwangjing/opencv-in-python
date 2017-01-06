import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_image():
    # '/f' is a special case in string, put a 'r' before the string to escape the '/f'
    # img = cv2.imread(r'C:\Users\jing\Desktop\flower.jpg', 1);
    global img

    # get the number of rows, cols and layers of the image
    shape = img.shape;
    print 'shape: ' + str(shape);
    # get the number of pixels
    size = img.size;
    print 'size: ' + str(size);
    # the data type of pixel
    pixel_type = img.dtype;
    print 'type: ' + str(pixel_type);
    # image.item can only return one value at a time, so need to pass in the BGR layer index.
    px = img.item(100, 100, 0);
    print 'old value: ' + str(px)
    img.itemset((100, 100, 0), 250);
    px = img.item(100, 100, 0);
    print 'new value: ' + str(px)


def get_roi():
    global img;
    area = img[250:360, 250:350];
    img[1:111, 1:101] = area;


# pass in the name of the window
def show_image(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL);
    cv2.imshow(window_name, img);
    cv2.setMouseCallback(window_name, get_pixel);
    k = cv2.waitKey(0) & 0xFF; # press '0', it will wait infinitely for a key stroke
    if k == ord('q'):
        cv2.destroyAllWindows();


def split_merge_image_channels():
    global img
    b, g, r = cv2.split(img);
    show_image('blue', b);
    show_image('green', g);
    show_image('red', r);
    img = cv2.merge((b, g, r));
    show_image('merged', img);

#cv2.copyMakeBorder(src,top,bottom,left,right,bordertype,dst=None, value=None);
def add_borders():
    len = 30
    # img1 = cv2.imread('C:\Users\jing\Desktop\opencv_logo.png', 1);
    img1 = cv2.imread(r'flower.jpg', 1);
    replicate = cv2.copyMakeBorder(img1, len, len, len, len, cv2.BORDER_REPLICATE);
    reflect = cv2.copyMakeBorder(img1, len, len, len, len, cv2.BORDER_REFLECT);
    reflect01 = cv2.copyMakeBorder(img1, len, len, len, len, cv2.BORDER_REFLECT101);
    wrap = cv2.copyMakeBorder(img1, len, len, len, len, cv2.BORDER_WRAP);
    constant = cv2.copyMakeBorder(img1, len, len, len, len, cv2.BORDER_CONSTANT, value=[255,0,0]);

    plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('origin')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
    plt.subplot(234), plt.imshow(reflect01, 'gray'), plt.title('reflect_01')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')

    plt.show();


def add_images() :
    x = np.uint8([255]);
    y = np.uint8([10]);
    print cv2.add(x, y); # return 255, as the max of uint8 is 255
    print x+y; # return (255 + 10) % 256 = 4;


def blend_image() :
    # cv2.addweighted(src, alpha, src2, beta, gamma, dst=none, dtype=none);
    # dst = alpha * src1 + beta * src2 + gamma
    img1 = cv2.imread('lena.png', 1);
    img2 = cv2.imread('opencv_logo.png', 1);
    dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0);
    show_image('blend', dst);


def get_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print img.item(x, y, 0);


def bitwise_image() :
    img2 = cv2.imread('opencv_logo.png', 1);
    img3 = cv2.imread('me before you.png', 1);

    rows, cols, channels = img2.shape; # get the size of logo.
    roi = img3 [0 : rows, 0 : cols]; # reserve the area in image3
    show_image('roi', roi);

    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY); # convert the logo to gray img, for threshold img.
    show_image('gray',img_gray);
    # First one is a retval. Second output is our thresholded image
    # cv2.threshold: (src[must gray scale], threshold,
    #  greater than threshold then set to 255, threshold type]
    ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY); # 255 is white, 0 is black
    show_image('mask', mask); # mask is white background, black content
    mask_inv = cv2.bitwise_not(mask); # get the inverse of the mask
    show_image('maskinv', mask_inv);

    # black out the area of logo in roi
    # src1 & mask = res1, src2 & mask = res2, res1 & res2
    # content is black, so the black logo is on roi.
    img3_bg = cv2.bitwise_and(roi, roi, mask=mask);
    print img3_bg.shape
    show_image('img3_bg', img3_bg);

    # take only region of logo from logo image
    # only the content of logo stayed, the background of logo is black.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv);
    show_image('img2_fg', img2_fg);

    # put  logo in roi and modify the main image
    # in img2_fg, background is black/0; in img3_bg, logo is black/0. use cv2.add() doesn't effect.
    dst = cv2.add(img3_bg, img2_fg)
    img3[0:rows, 0:cols] = dst;
    show_image('dst', dst);
    show_image('result', img3);


def main():
    # show_image('old', img)
    # get_image();
    # get_roi();
    # show_image('new', img);
    # split_merge_image_channels();
    # add_borders();
    # blend_image();
    bitwise_image();


# cv2.getTickCount() returns the number of clock-cycles after a reference event (like the moment machine was
# switched ON) to the moment this function is called. So if you call it before and after the function execution, you get
# number of clock-cycles used to execute a function.
# cv2.getTickFrequency function returns the frequency of clock-cycles, or the number of clock-cycles per second. So
# to find the time of execution in seconds.

e1 = cv2.getTickCount();
img = cv2.imread('lena.png', 1);
main();
e2 = cv2.getTickCount();
time = (e2 - e1) / cv2.getTickFrequency();
print 'time: '+str(time); # time is second unit.

