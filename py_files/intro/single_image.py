import cv2 as cv

gpu_frame = cv.cuda_GpuMat()

screenshot = cv.imread('../../media/drip.png')
gpu_frame.upload(screenshot)

screenshot = cv.cuda.cvtColor(gpu_frame, cv.COLOR_RGB2BGR)
screenshot = cv.cuda.resize(screenshot, (400, 400))

screenshot = screenshot.download()

# display image
from PIL import Image
Image.fromarray(screenshot)  # Image.fromarray(screenshot).save('../../media/resized_drip.png')
