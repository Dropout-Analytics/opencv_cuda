import cv2 as cv
import numpy as np


def preprocess(video):
    # init video capture with video
    vod = cv.VideoCapture(video)

    # get video FPS & total number of frames
    fps = vod.get(cv.CAP_PROP_FPS)
    n_frames = vod.get(cv.CAP_PROP_FRAME_COUNT)

    # read the first frame
    ret, frame = vod.read()

    scale = 0.25

    # was first frame read successfully? (ret == True)
    # if ret:

    # create GPU matrix (picture frame)
    gpu_frame = cv.cuda_GpuMat()
    
    # as long as there are more frames
    while ret:

        # upload this frame to GPU
        gpu_frame.upload(frame)

        # do stuff
        resized = cv.cuda.resize(gpu_frame, (int(1280 * scale), int(720 * scale)))

        luv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2LUV)
        
        hsv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2HSV)

        gray = cv.cuda.cvtColor(resized, cv.COLOR_BGR2GRAY)
        
        thresh = cv.cuda.threshold(gray, 155, 255, cv.THRESH_BINARY)
        
        canny = cv.Canny(gray.download(), 155, 255)


        # convert gray & canny to 3d arrays (so they can be dislayed with colored arrays)
        gray = cv.cuda.cvtColor(gray, cv.COLOR_GRAY2BGR)
        thresh = cv.cuda.cvtColor(thresh[1], cv.COLOR_GRAY2BGR)
        canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)  # canny on CPU

        # download new image(s) from GPU to CPU
        resized = resized.download()
        luv = luv.download()
        hsv = hsv.download()
        gray = gray.download()
        thresh = thresh.download()

        # visualization
        top_row = np.concatenate((resized, gray), axis=1)
        middle_row = np.concatenate((luv, thresh), axis=1)
        bottom_row = np.concatenate((hsv, canny), axis=1)

        joined = np.concatenate((top_row, middle_row, bottom_row), axis=0)
        
        cv.imshow('OG | GRAY | LUV | THRESH | HSV | CANNY', joined)

        k = cv.waitKey(1)
        # user Esc
        if k == 27:
            break

        # continue to next frame
        ret, frame = vod.read()


    # release the capture
    vod.release()

    # destroy all windows
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    video = '../../media/corn.mp4'
    preprocess(video)
