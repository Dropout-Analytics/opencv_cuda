import cv2 as cv
import numpy as np


def preprocess(video):
    # init video capture with video
    vod = cv.VideoCapture(video)

    # read the first frame
    ret, frame = vod.read()

    scale = 0.5

    # create GPU matrix (picture frame)
    gpu_frame = cv.cuda_GpuMat()

    # as long as frames are successfully read
    while ret:

        # upload this frame to GPU
        gpu_frame.upload(frame)

        # do stuff
        resized = cv.cuda.resize(gpu_frame, (int(1280 * scale), int(720 * scale)))

        luv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2LUV)
        
        hsv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2HSV)

        gray = cv.cuda.cvtColor(resized, cv.COLOR_BGR2GRAY)
        

        # convert gray & canny to 3d arrays (so they can be dislayed with colored arrays)
        gray = cv.cuda.cvtColor(gray, cv.COLOR_GRAY2BGR)

        # download new image(s) from GPU to CPU
        resized = resized.download()
        luv = luv.download()
        hsv = hsv.download()
        gray = gray.download()

        # visualization
        top_row = np.concatenate((resized, luv), axis=1)
        bottom_row = np.concatenate((hsv, gray), axis=1)

        joined = np.concatenate((top_row, bottom_row), axis=0)
        
        cv.imshow('OG | LUV | HSV | GRAY', joined)

        k = cv.waitKey(1)
        # user Esc
        if k == 27:
            break

        # continue to next frame
        ret, frame = vod.read()

    # release the capture & destroy all windows
    vod.release()
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    video = '../../media/corn.mp4'
    preprocess(video)
