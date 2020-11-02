import cv2 as cv
import dask.delayed
from dask import compute

img_files = ['bear.png', 'drip.png', 'tldr.png', 'frog.png']
img_files_2 = ['apple.png', 'eye.png', 'window.png', 'blinds.png']


@dask.delayed
def preprocess(files):
    # copy image files
    i_files = files.copy()
    
    # create GPU frame to hold images
    gpu_frame = cv.cuda_GpuMat()
    
    for i in range(len(i_files)):
        # load image (CPU)
        screenshot = cv.imread(f'../../media/{i_files[i]}')

        # fit screenshot to (GPU) frame
        gpu_frame.upload(screenshot)

        # translate colors to opencv (numpy.ndarray -> cv2.cuda_GpuMat)
        screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
        screenshot = cv.cuda.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

        # inverted threshold @ 100
        screenshot = cv.cuda.threshold(screenshot, 125, 255, cv.THRESH_BINARY)

        # resize image
        screenshot = cv.cuda.resize(screenshot[1], (200, 200))
        
        # download image from GPU (cv2.cuda_GpuMat -> numpy.ndarray)
        screenshot = screenshot.download()

        # replace file name with new image
        i_files[i] = screenshot
    
    # output preprocessed images
    return i_files


# do the delayed
set_a = dask.delayed(preprocess)(img_files)
set_b = dask.delayed(preprocess)(img_files_2)
out_a, out_b = compute(*[set_a, set_b])
    
# combine both sets of 4 into 1 image
import numpy as np

top_left = np.concatenate((out_a[0], out_a[2]), axis=0)
top_right = np.concatenate((out_b[0], out_b[2]), axis=0)
bottom_left = np.concatenate((out_a[1], out_a[3]), axis=0)
bottom_right = np.concatenate((out_b[1], out_b[3]), axis=0)

top_row = np.concatenate((top_left, top_right), axis=1)
bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)

big_image = np.concatenate((top_row, bottom_row), axis=1)

# display inline with PIL
from PIL import Image
Image.fromarray(big_image)  # Image.fromarray(big_image).save('../../media/delayed_big_image.png')
