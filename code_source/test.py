import pyarrow.parquet as pq
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageChops
from numpy import asarray
from copy import deepcopy
from matplotlib import pyplot
from trace_vecteurs import draw_head_pose

df = pd.read_parquet('series.pq', engine='pyarrow')


list_a = []
list_b = []
list_c = []

for i in list(df.index):
    row = df.loc[i]
    dict_a = {}
    dict_b = {}
    dict_c = {}
    dict_a['rotation-x'] = row[0]
    dict_a['rotation-y'] = row[1]
    dict_a['rotation-z'] = row[2]
    dict_a['translation-x'] = row[3]
    dict_a['translation-y'] = row[4]
    dict_a['translation-z'] = row[5]

    dict_b['rotation-x'] = row[6]
    dict_b['rotation-y'] = row[7]
    dict_b['rotation-z'] = row[8]
    dict_b['translation-x'] = row[9]
    dict_b['translation-y'] = row[10]
    dict_b['translation-z'] = row[11]

    dict_c['rotation-x'] = row[12]
    dict_c['rotation-y'] = row[13]
    dict_c['rotation-z'] = row[14]
    dict_c['translation-x'] = row[15]
    dict_c['translation-y'] = row[16]
    dict_c['translation-z'] = row[17]

    list_a.append(dict_a)
    list_b.append(dict_b)
    list_c.append(dict_c)




# on trace les trois cubes sur chaque image


path = 'vid0/frames_vid0_test'

for i in range(0,300):
  
  image = Image.open('vid0/frames_vid0/frame'+str(i)+'.jpg')

  data = asarray(image)

  data = draw_head_pose(list_a[i],data,color=(255, 0, 0)) # BLUE
  data = draw_head_pose(list_b[i],data,color=(0, 255, 0)) #GREEN
  data = draw_head_pose(list_c[i],data,color=(0, 0, 255)) #RED

  img = asarray(data)

  
  cv2.imwrite(os.path.join(path , 'frame'+str(i)+'.jpg'), img)


path = 'vid1/frames_vid1_test'

for i in range(0,300):
  
  image = Image.open('vid1/frames_vid1/frame'+str(i)+'.jpg')

  data = asarray(image)

  data = draw_head_pose(list_a[i],data,color=(255, 0, 0))  #BLUE
  data = draw_head_pose(list_b[i],data,color=(0, 255, 0)) #GREEN 
  data = draw_head_pose(list_c[i],data,color=(0, 0, 255)) #RED

  img = asarray(data)

  
  cv2.imwrite(os.path.join(path , 'frame'+str(i)+'.jpg'), img)


path = 'vid2/frames_vid2_test'

for i in range(0,300):
  
  
  image = Image.open('vid2/frames_vid2/frame'+str(i)+'.jpg')

  data = asarray(image)

  data = draw_head_pose(list_a[i],data,color=(255, 0, 0)) #BLUE
  data = draw_head_pose(list_b[i],data,color=(0, 255, 0)) #GREEN
  data = draw_head_pose(list_c[i],data,color=(0, 0, 255)) #RED

  img = asarray(data)

  
  cv2.imwrite(os.path.join(path , 'frame'+str(i)+'.jpg'), img)


