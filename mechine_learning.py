import cv2
import numpy as np
import matplotlib.pyplot as plt
import show_image as imgHelper;

# ORC: optical character recognition
def knn_test():
    # feature set containing (x,y) values of 25 known/training data
    # randint(min, max, size)
    # generate a 25*2 matrix in float32 and in range [0, 100), as coordinates
    trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32);

    # labels each one either red or blue with numbers 0 and 1
    # generate 25 points, 0 for blue, 1 for red.
    response = np.random.randint(0, 2, (25, 1)).astype(np.float32);

    # take red families and plot them
    red = trainData[response.ravel() == 0];
    print "red: "
    print trainData[response.ravel() == 0];
    # scatter(x, y, scalar, color, marker_style);
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^');

    # take blue families and plot them
    blue = trainData[response.ravel() == 1];
    print "blue: "
    print response.ravel() == 1;
    print trainData[response.ravel() == 1];
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's');

    # create a new point
    newcommer = np.random.randint(0, 100, (1, 2)).astype(np.float32);
    plt.scatter(newcommer[:, 0], newcommer[:, 1], 80, 'g', 'o');

    # generate another 10 points, to find neighbors for each of them.
    newcommers = np.random.randint(0, 100, (10, 2)).astype(np.float32);
    plt.scatter(newcommers[:, 0], newcommers[:, 1], 80, 'y', 'o');

    knn = cv2.ml.KNearest_create();
    knn.train(trainData, cv2.ml.ROW_SAMPLE, response);
    ret, results, neighbours, dist = knn.findNearest(newcommers, 3); # choose 3 nearest neighbors

    print "result: ", results, "\n"
    print "neighbours: ", neighbours, "\n"
    print "distance: ", dist

    plt.show()


def ocr_knn_test():
    img = cv2.imread('digits.png');
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    imgHelper.show_image('origin img', gray);
    print "size of origin ", gray.shape; # (1000, 2000)

    # now split the image to 5000 cells, each 20*20 size
    # cut 1000 rows into 50 pieces, 2000 cols into 100 pieces
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)];
    print "cells shape ", cells.__sizeof__();

    # make it into a numpy array. it size will be (50, 100, 20, 20)
    # array has 50 rows, 100 cols, each element is 20*20.
    x = np.array(cells);
    print "shape ", x.shape;

    # prepare train_data and test_data, left 50 cols are train, right cols are test
    train = x[:, :50].reshape(-1, 400).astype(np.float32); # size = (2500, 400)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32); # size = (2500, 400)

    # create labels for train and test data
    k = np.arange(10); # 1 - 9
    train_labels = np.repeat(k, 250)[:, np.newaxis]; # size = [2500, 1], 250 per digit.
    test_labels = train_labels.copy();

    # # save the data
    # np.savez('knn_data.npz', train = train, train_labels = train_labels);
    # # load data
    # with np.load('knn_data.npz') as data:
    #     print data.files
    #     train = data['train'];
    #     train_labels = data['train_labels']

    # initiate KNN, train the data, then test it with test data for k = 1
    knn = cv2.ml.KNearest_create();
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels);
    ret, results, neighbors, dist = knn.findNearest(test, k=5);
    # for i in xrange(0, 2500):
    #     print "result : ", results[i];

    # now we check the accuracy of classification
    # compare the result with test_labels and check which are wrong
    matches = results == test_labels;
    correct = np.count_nonzero(matches);
    accuracy = correct * 100.0 / results.size;
    print accuracy


def orc_knn_alphabet():
    # load the data, converters convert the letter to a number
    data = np.loadtxt('C:\opencv\sources\samples\data\letter-recognition.data',
                      dtype = 'float32', delimiter=',',
                      converters= {0: lambda  ch: ord(ch)-ord('A')})
    print "data size: ", data.shape # (20000, 17)

    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data, 2);

    # split trainData and testData to feature and responses
    responses, trainData = np.hsplit(train, [1]);
    labels, testData = np.hsplit(test, [1]);

    # initiate the knn, classify, measure accuracy
    knn = cv2.ml.KNearest_create();
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses);
    ret, result, neighbors, dist = knn.findNearest(testData, k=5);

    correct = np.count_nonzero(result == labels);
    accuracy = correct * 100.0 / 10000

    print accuracy


def main():
    # knn_test();
    # ocr_knn_test();
    orc_knn_alphabet();


main();