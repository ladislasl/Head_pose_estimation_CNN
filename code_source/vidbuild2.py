import cv2
import numpy as np
import glob

frameSize = (500, 500)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

for filename in glob.glob('vid0/vid0_frames_test/*.jpg'):
    img = cv2.imread(filename)
    out.write(img)

out.release()