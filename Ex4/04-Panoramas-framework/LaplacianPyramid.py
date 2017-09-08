import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal




# some helper functions (internally, images are represented as floats to avoid errors)
def LoadImage(filename):
	return mpimg.imread(filename).astype(np.float)

def SaveImage(filename, img):
	plt.imsave(filename, np.clip(img, 0, 255).astype(np.uint8))

def ShowImage(title, img):
	plt.figure(title)
	assert img.dtype == np.float
	plt.imshow(np.clip(img, 0, 255).astype(np.uint8))


def Down(img, n=1):
	assert img.dtype == np.float
	down = None
	# ------
	# Your code...
	kernel = 0.25*np.array([[1,1],[1,1]])
	down = img
	for _ in range(n):
		filtered = np.stack([scipy.signal.convolve(down[:,:,ch], kernel, mode='same') for ch in range(3)], axis=2) 
		down = filtered[::2,::2,:]
		assert down.shape == (filtered.shape[0]/2, filtered.shape[1]/2, 3)
	# ------
	return down


def Up(img, n=1):
	assert img.dtype == np.float
	Up = None
	# ------
	# Your code...
	sh = img.shape
	for _ in range(n):
		Up = np.empty((img.shape[0]*2, img.shape[1]*2, 3))
		Up[::2,::2,:] = img
		
		## unknown values
		x = [i for i in range(1, Up.shape[0],2)]
		## known values 
		## shape[0] = shape[1]
		xp = [i for i in range(0, Up.shape[0],2)]

		## interpolate in the first dimension for each channel
		for i in range(Up.shape[0]):
			for ch in range(3):
				Up[i, 1::2, ch] = np.interp(x, xp, Up[i, ::2, ch])

		## interpolate in the second dimension for each channel
		for j in range(Up.shape[1]):
			for ch in range(3):
				Up[1::2, j, ch] = np.interp(x, xp, Up[::2, j, ch])
		img = Up
	# ------
	return Up


# Computes the Gaussian Pyramid from an image
def GaussianPyramid(img):
	assert img.dtype == np.float
	assert img.shape[0] == img.shape[1] # check for square
	s = img.shape[0]
	assert ((s & (s - 1)) == 0) and s > 0 # check power of two

	gaussianPyramid = None
	# ------
	# Your code...
	gaussianPyramid = [Down(img, i) for i in range(10)]
	# ------
	return gaussianPyramid

# Computes the Laplacian pyramid from a Gaussian pyramid
def LaplacianPyramid(gaussianPyramid):
	laplacianPyramid = None
	# ------
	# Your code...
	laplacianPyramid = [gaussianPyramid[i] - Up(gaussianPyramid[i+1]) for i in range(len(gaussianPyramid)-1)]
	laplacianPyramid.append(gaussianPyramid[len(gaussianPyramid)-1])
	# ------
	return laplacianPyramid

def Decompose(img):
	return LaplacianPyramid(GaussianPyramid(img))

def Reconstruct(pyramid):
	reconstruction = None
	# ------ 
	# Your code...
	levels = len(pyramid)-1
	reconstruction = pyramid[levels]
	for level in reversed(range(levels)):
		reconstruction = Up(reconstruction) + pyramid[level]

	# ------
	return reconstruction


def BlendImage(img1, img2, mask):
	blended = None
	# ------ 
	# Your code...
	## build Laplacian pyramids for both images and the mask
	laplacian1 = Decompose(img1)
	laplacian2 = Decompose(img2)
	gmask = GaussianPyramid(mask)

	assert len(laplacian1) == len(laplacian2) 
	
	levels = len(laplacian1)
	blended = np.zeros(laplacian1[levels-1].shape)

	for level in reversed(range(1,levels)):
		blended += laplacian1[level]*gmask[level] + laplacian2[level]*(1-gmask[level])
		blended = Up(blended)
	# ------
	return blended



# The main program
img = LoadImage("input.jpg")
ShowImage("Original", img)

down = Down(img, 5)
ShowImage("Downsampled", down)

up = Up(down, 5)
ShowImage("Upsampled", up)


p = Decompose(img)
# for i in range(len(p)):
# 	ShowImage("Lvl {0}".format(i), p[i]+128)

r = Reconstruct(p)
ShowImage("Reconstruction", r)

beach1 = LoadImage("beach2.jpg")
beach2 = LoadImage("beach1.jpg")
mask = LoadImage("BeachMask.png")
b = BlendImage(beach1, beach2, mask)
ShowImage("Blended", b)

# also do a naive blending:
naive = beach1*mask + (np.ones(mask.shape)-mask)*beach2
ShowImage("naive", naive)

SaveImage("blended.png", b)
SaveImage("naive.png", naive)

plt.show()
