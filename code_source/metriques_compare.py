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
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2


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


def compare(im1,im2):
  
  # convert the images to grayscale
  grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  (score, diff) = compare_ssim(grayA, grayB, full=True)
  return score



#comparaison métrique

nombre_de_matchvid0=[0,0,0]

for i in range(0,300):

  image = cv2.imread('vid0/frames_vid0/frame'+str(i)+'.jpg')
  data = asarray(image) 
  data = draw_head_pose(list_a[i],data,color=(255, 255, 255))
  original = cv2.imread('vid0/frames_vid0_traced/frame'+str(i)+'.jpg')
  
  
  if compare(data, original)>0.97:
     nombre_de_matchvid0[0] += 1

  image = cv2.imread('vid0/frames_vid0/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_b[i],data,color=(255, 255, 255))
  original = cv2.imread('vid0/frames_vid0_traced/frame'+str(i)+'.jpg')

 
  if compare(data, original)>0.97:
     nombre_de_matchvid0[1] += 1

  image = cv2.imread('vid0/frames_vid0/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_c[i],data,color=(255, 255, 255))
  original = cv2.imread('vid0/frames_vid0_traced/frame'+str(i)+'.jpg')


  if compare(data, original)>0.97:
     nombre_de_matchvid0[2] += 1



max=0
index_max=0
for i in range(3):
    if nombre_de_matchvid0[i] > max:
        max=nombre_de_matchvid0[i]
        index_max=i

if index_max ==0:
    corresp='a'
if index_max ==1:
    corresp='b'
if index_max ==2:
    corresp='c'


print(nombre_de_matchvid0)
print('video 0 correspond aux vecteurs :'+ corresp)





nombre_de_matchvid1=[0,0,0]

for i in range(0,300):

  image = cv2.imread('vid1/frames_vid1/frame'+str(i)+'.jpg')
  data = asarray(image) 
  data = draw_head_pose(list_a[i],data,color=(255, 255, 255))
  original = cv2.imread('vid1/frames_vid1_traced/frame'+str(i)+'.jpg')
  

  if compare(data, original)>0.97:
     nombre_de_matchvid1[0] += 1

  image = cv2.imread('vid1/frames_vid1/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_b[i],data,color=(255, 255, 255))
  original = cv2.imread('vid1/frames_vid1_traced/frame'+str(i)+'.jpg')


  if compare(data, original)>0.97:
     nombre_de_matchvid1[1] += 1

  image = cv2.imread('vid1/frames_vid1/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_c[i],data,color=(255, 255, 255))
  original = cv2.imread('vid1/frames_vid1_traced/frame'+str(i)+'.jpg')

 
  if compare(data, original)>0.97:
     nombre_de_matchvid1[2] += 1


max=0
index_max=0
for i in range(3):
    if nombre_de_matchvid1[i] > max:
        max=nombre_de_matchvid1[i]
        index_max=i

if index_max ==0:
    corresp='a'
if index_max ==1:
    corresp='b'
if index_max ==2:
    corresp='c'


print(nombre_de_matchvid1)
print('video 1 correspond aux vecteurs :'+ corresp)





nombre_de_matchvid2=[0,0,0]

for i in range(0,300):

  image = cv2.imread('vid2/frames_vid2/frame'+str(i)+'.jpg')
  data = asarray(image) 
  data = draw_head_pose(list_a[i],data,color=(255, 255, 255))
  
  original = cv2.imread('vid2/frames_vid2_traced/frame'+str(i)+'.jpg')
  

  if compare(data, original)>0.97:
     nombre_de_matchvid2[0] += 1

  image = cv2.imread('vid2/frames_vid2/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_b[i],data,color=(255, 255, 255))

  original = cv2.imread('vid2/frames_vid2_traced/frame'+str(i)+'.jpg')

  

  if compare(data, original)>0.97:
     nombre_de_matchvid2[1] += 1

  image = cv2.imread('vid2/frames_vid2/frame'+str(i)+'.jpg')
  data = asarray(image)
  data = draw_head_pose(list_c[i],data,color=(255, 255, 255))

  original = cv2.imread('vid2/frames_vid2_traced/frame'+str(i)+'.jpg')

  
  if compare(data, original)>0.97:
     nombre_de_matchvid2[2] += 1



max=0
index_max=0
for i in range(3):
    if nombre_de_matchvid2[i] > max:
        max=nombre_de_matchvid2[i]
        index_max=i

if index_max ==0:
    corresp='a'
if index_max ==1:
    corresp='b'
if index_max ==2:
    corresp='c'


print(nombre_de_matchvid2)
print('video 2 correspond aux vecteurs :'+ corresp)

