import cv2
import numpy as np


###################################
		### opencv ###
###################################

## read a pic
img = cv2.imread('bonn.jpg')
cv2.imshow('img', img)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow('img', img_rgb)

## convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img_gray)

## apply a laplacian filter of size 7
img_laplace = cv2.Laplacian(img_gray, ddepth=-1, ksize=7)
cv2.imshow('laplacian', img_laplace)

## write a png 
cv2.imwrite("laplace.png", img_laplace)

## apply a sepia filter
S = np.array( [[0.393, 0.769, 0.189],
	[0.349, 0.686, 0.168],
	[0.272, 0.534, 0.131]])

# norms = np.sum(S, axis = 1)
norms = np.linalg.norm(S, axis = 1)

S[0,:] /= norms[0]
S[1,:] /= norms[1]
S[2,:] /= norms[2]

S = np.fliplr(S)
img_sepia = np.array([[np.flipud(S.dot(img[i,j,:])) for j in range(img.shape[1])] for i in range(img.shape[0])])
cv2.imshow('sepia', img_sepia.astype(np.uint8)) 

# img_sepia = np.array([[S.dot(img_rgb[i,j,:]) for j in range(img.shape[1])] for i in range(img.shape[0])])
# img_sepia = cv2.cvtColor(img_sepia.astype(np.uint8), cv2.COLOR_RGB2BGR)
# cv2.imshow('sepia', img_sepia)


## final things
cv2.waitKey(0)
cv2.destroyAllWindows()


###################################
	### linear systems ###
###################################

import scipy.sparse as ss
import scipy.sparse.linalg as ssl

A = np.array([[1,2,3], [1,5,6], [2,2,3]])
b = np.array([[1], [1], [3]])

x = np.linalg.solve(A, b)
print(x)

n = 1000
d = np.repeat(1, n)
d1 = np.repeat(2, n)

B = ss.diags([d,d1,d], [0,1,2])
c = list(range(n))

y = ssl.spsolve(B,c)

B_full = B.toarray()
z = np.linalg.solve(B_full, c)