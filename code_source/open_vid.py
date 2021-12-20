import cv2
vidcap = cv2.VideoCapture('samples/sample-2-traced.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames_vid2_traced/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
