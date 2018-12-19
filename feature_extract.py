import cv2
import numpy as np
import matplotlib as plt
import cv_algorithms
import scipy.io as sc

img = cv2.imread('two.jpg',0)
h = img.shape[0]
w = img.shape[1]
for x in range (0,h):
	for y in range (0,w):
		if img[x,y] >= 100:
			img[x,y] = 255
		else:
			img[x,y] = 0
cv2.imshow('binary',img)
zhang_suen = cv_algorithms.zhang_suen(img)
cv2.imshow('thin',zhang_suen)

new = cv2.resize(zhang_suen,(40,40),interpolation=cv2.INTER_AREA)
for x in range (0,40):
	for y in range (0,40):
		if new[x,y] != 0:
			new[x,y] = 255;

zone1 = new[0:10,0:10]
zone2 = new[10:20,0:10]
zone3 = new[20:30,0:10]
zone4 = new[30:40,0:10]

zone5 = new[0:10,10:20]
zone6 = new[10:20,10:20]
zone7 = new[20:30,10:20]
zone8 = new[30:40,10:20]

zone9 = new[0:10,20:30]
zone10 = new[10:20,20:30]
zone11 = new[20:30,20:30]
zone12 = new[30:40,20:30]

zone13 = new[0:10,30:40]
zone14 = new[10:20,30:40]
zone15 = new[20:30,30:40]
zone16 = new[30:40,30:40]

def calculate_feature(zone):
	value = np.zeros((19))
	k = 0
	for i in range(1,10):
		n1 = zone.diagonal(i)
		n2 = zone.diagonal(-i)
		s1 = sum(n1)
		s2 = sum(n2)
		value[k] = s1
		value[k+1] = s2
		k = k + 2
	value[18] = sum(zone.diagonal(0))
	#print(value)
	feature_value = np.mean(value)
	print(feature_value)
	return feature_value

feature_vector = np.zeros((4,4))

feature_vector[0,0] = calculate_feature(zone1)
feature_vector[0,1] = calculate_feature(zone2)
feature_vector[0,2] = calculate_feature(zone3)
feature_vector[0,3] = calculate_feature(zone4)

feature_vector[1,0] = calculate_feature(zone5)
feature_vector[1,1] = calculate_feature(zone6)
feature_vector[1,2] = calculate_feature(zone7)
feature_vector[1,3] = calculate_feature(zone8)


feature_vector[2,0] = calculate_feature(zone9)
feature_vector[2,1] = calculate_feature(zone10)
feature_vector[2,2] = calculate_feature(zone11)
feature_vector[2,3] = calculate_feature(zone12)


feature_vector[3,0] = calculate_feature(zone13)
feature_vector[3,1] = calculate_feature(zone14)
feature_vector[3,2] = calculate_feature(zone15)
feature_vector[3,3] = calculate_feature(zone16)

print(feature_vector)

#n1 = cv2.resize(feature_vector,(400,400),interpolation=cv2.INTER_CUBIC)
cv2.imshow('img',new)
#cv2.imshow('img',n1)
cv2.waitKey(0)
cv2.destroyAllWindows()
 

