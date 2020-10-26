import cv2 as cv

img_files = ['bear.png', 'drip.png', 'tldr.png', 'frog.png']

# create frame to hold images (cv2.cuda_GpuMat)
gpu_frame = cv.cuda_GpuMat()

for i in range(len(img_files)):
    # load image (CPU)
    screenshot = cv.imread(f"../../media/{img_files[i]}")

    # fit screenshot to (GPU) frame
    gpu_frame.upload(screenshot)
    
    # translate colors to opencv (numpy.ndarray -> cv2.cuda_GpuMat)
    screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
    
    # inverted threshold @ 100
    screenshot = cv.cuda.threshold(screenshot, 105, 255, cv.THRESH_BINARY_INV)
        
    # resize image
    screenshot = cv.cuda.resize(screenshot[1], (200, 200))

    # download image from GPU (cv2.cuda_GpuMat -> numpy.ndarray)
    screenshot = screenshot.download()
    
    # replace file name with new image
    img_files[i] = screenshot

    
# combine all 4 pictures into 1 image
import numpy as np
top_row = np.concatenate((img_files[0], img_files[1]), axis=1)
bottom_row = np.concatenate((img_files[2], img_files[3]), axis=1)
big_image = np.concatenate((top_row, bottom_row), axis=0)

# display inline with PIL
from PIL import Image
Image.fromarray(big_image)  # Image.fromarray(big_image).save('../../media/big_image.png')
