import numpy as np
import cv2


def turnOnCamera():
        # pass in the device number
        cap = cv2.VideoCapture(0);
        print cap.get(3); # get the width of frame, in this case 640.0
        print cap.get(4); # get the height of frame, 480.0

        ret = cap.set(3, 320); # set the frame width to 320.0
        ret = cap.set(4, 240); # set the frame height to 240.0

        # use cap.isOpened() to check cap has been initialized or not.
        # if cap is not on, use cap.open() to open the cap.
        while (cap.isOpened()):
            # capture frame by frame,
            # returns boolean value, if frame is read correctly, it will be True
            ret, frame = cap.read();

            # our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

            # display the resulting frame in gray
            cv2.imshow('frame', gray);

            # press 'q' to exit the video
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break;

        # when everything is done, release the capture.
        cap.release();
        cv2.destroyAllWindows();


def displayVideo() :
    cap = cv2.VideoCapture('C:\Users\jing\Videos\FUNNY\bb1.mp4'); # actually it failed...
    print cap.isOpened(); # false

    while (cap.isOpened()) :
        ret, frame = cap.read();

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

        cv2.imshow(frame, gray);
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    cap.release();
    cv2.destroyAllWindows()


def saveVideo() :
    cap = cv2.VideoCapture(0);

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID');
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    while (cap.isOpened()) :
        ret, frame = cap.read();
        if ret == True :
            frame = cv2.flip(frame, 0);

            # write the flipped frame
            out.write(frame);

            cv2.imshow('frame', frame);
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break;
        else :
            break;

    cap.release();
    out.release();
    cv2.destroyAllWindows();


def main() :
    turnOnCamera();
    displayVideo();
    saveVideo();

main();