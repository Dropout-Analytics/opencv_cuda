import cv2 as cv
import numpy as np


def preprocess(video):
    # init video capture with video
    vod = cv.VideoCapture(video)

    # get video FPS & total number of frames
    fps = vod.get(cv.CAP_PROP_FPS)
    n_frames = vod.get(cv.CAP_PROP_FRAME_COUNT)

    out = cv.VideoWriter('../../media/combo_output.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, (540, 640))

    # read the first frame
    ret, frame = vod.read()
    i=0

    # was first frame read successfully? (ret == True)
    if ret:

        # create GPU matrix (picture frame)
        gpu_frame = cv.cuda_GpuMat()
        
        # as long as there are more frames
        while True:

            # upload this frame to GPU
            gpu_frame.upload(frame)

            # do stuff
            try:

                resized = cv.cuda.resize(gpu_frame, (int(1280 * 0.25), int(720 * 0.25)))

                luv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2LUV)
                
                hsv = cv.cuda.cvtColor(resized, cv.COLOR_BGR2HSV)

                gray = cv.cuda.cvtColor(resized, cv.COLOR_BGR2GRAY)
                
                thresh = cv.cuda.threshold(gray, 115, 255, cv.THRESH_BINARY)
                
                canny = cv.Canny(gray.download(), 115, 255)


                # convert gray & canny to 3d arrays (so they can be dislayed with colored arrays)
                gray = cv.cuda.cvtColor(gray, cv.COLOR_GRAY2BGR)
                thresh = cv.cuda.cvtColor(thresh[1], cv.COLOR_GRAY2BGR)
                canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

                # download new image(s) from GPU to CPU
                resized = resized.download()
                luv = luv.download()
                hsv = hsv.download()
                gray = gray.download()
                thresh = thresh.download()
        
                # visualization
                top_row = np.concatenate((resized, thresh), axis=1)
                middle_row = np.concatenate((hsv, gray), axis=1)
                bottom_row = np.concatenate((luv, canny), axis=1)

                joined = np.concatenate((top_row, middle_row, bottom_row), axis=0)
                
                cv.imshow('OG | GRAY | LUV | THRESH | HSV | CANNY', joined)
                out.write(joined)

                k = cv.waitKey(1)
                # user Esc
                if k == 27:
                    break

                # continue to next frame
                ret, frame = vod.read()

            except:
                break
    
    # release the capture
    vod.release()

    # destroy all windows
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    video = '../../media/ankles.mp4'
    preprocess(video)
