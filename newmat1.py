import cv2
import numpy as np
import matplotlib as plt
import cv_algorithms
import scipy.io as sc
import math 

img = cv2.imread('three.jpg',0)
output = np.zeros((10))
output[3] = 1
print(output)
h = img.shape[0]
w = img.shape[1]
for x in range (0,h):
	for y in range (0,w):
		if img[x,y] >= 100:
			img[x,y] = 0
		else:
			img[x,y] = 255
cv2.imshow('binary',img)
zhang_suen = cv_algorithms.zhang_suen(img)
cv2.imshow('thin',zhang_suen)

new = cv2.resize(zhang_suen,(40,40),interpolation=cv2.INTER_AREA)
for x in range (0,40):
	for y in range (0,40):
		if new[x,y] != 0:
			new[x,y] = 255;
new = new.astype(np.float64)
for i in range(0,40):
	for j in range(0,40):
		new[i,j] = (new[i,j]/255)


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
	feature_value = np.mean(value)/255
	#print(feature_value)
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

#print(feature_vector.ravel())
features1 = feature_vector.ravel()
#print(img1)
#cv2.imshow('im',new)


#feed forward................
features = np.zeros((17))
features[0] = 1
for i in range (0,16):
	features[i+1] = features1[i]
features = features.astype(np.float64)
#print(features)

theta1 = sc.loadmat('newtheta1.mat')
theta2 = sc.loadmat('newtheta2.mat')

theta1 = theta1['Theta1']
theta2 = theta2['theta2']

print("")
print(theta1[0])

def sigmoid(z):
	k = math.exp((-z))
	sig = 1/(1+k)
	#print(sig)
	return sig

def sigmoid_derivative(z):
	a = sigmoid(z)
	b = 1 - a
	c = a*b
	return c

def call_sigmoid(features,theta,dim1,dim2):
	a_21 = np.zeros((dim1))
	a_2 = a_21.astype(np.float64)
	
	#a_2[0] = 0.00043434
	for i in range(0,dim1):
		for j in range(0,dim2):
			a_2[i] = a_2[i] + (features[j] * theta[i][j]) 
		a_2[i] = sigmoid(a_2[i])
	return a_2
def call_sigmoid_derivative(features,theta,dim1,dim2):
	a_21 = np.zeros((dim1))
	a_2 = a_21.astype(np.float64)
	
	
	for i in range(0,dim1):
		for j in range(0,dim2):
			a_2[i] = a_2[i] + (features[j] * theta[i][j]) 
		a_2[i] = sigmoid_derivative(a_2[i])
	return a_2

print("activation of layer 2")
layer2_activation1 = call_sigmoid(features,theta1,25,17)

layer2_activation = np.ones((26))
for i in range (0,25):
	layer2_activation[i+1] = layer2_activation1[i]
layer2_activation = layer2_activation.astype(np.float64)

#print("activation of layer 3")
layer3_activation = call_sigmoid(layer2_activation,theta2,10,26)


#########Back-Propagation

#delta for output layer 
delta3 = np.zeros((10))
for i in range(0,10):
	delta3[i] = layer3_activation[i] - output[i]
#print(delta3)


#delta for layer 2
theta2_transpose = theta2.transpose()

mult = np.matmul(theta2_transpose,delta3)
#print(mult)


layer2_derivative1 = call_sigmoid_derivative(features,theta1,25,17)

layer2_derivative = np.ones((26))
for i in range (0,25):
	layer2_derivative[i+1] = layer2_derivative1[i]
layer2_derivative = layer2_derivative.astype(np.float64)

delta2 = np.zeros((26))
for i in range(0,26):
	delta2[i] = (mult[i]*layer2_derivative[i])
print("delta2.............")
print(delta2)

cv2.waitKey(0)
cv2.destroyAllWindows()
