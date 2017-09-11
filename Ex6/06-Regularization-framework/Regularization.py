import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve, correlate
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsmr

plt.close('all')




# load image, crop it and take the red channel:
img = plt.imread("Lenna.png")[200:300,200:320,0]
#img = plt.imread("Lenna.png")[250:280,250:290,0]
imgShape = img.shape
pix = imgShape[0]*imgShape[1] # number of pixels

# kernels
# ---------
# your code
# C = ...
# d_x = ...
# d_y = ...
# ---------


l = 0.001
rho = 0.1 # strength of regularization
sigma = 0.01 # the amount of noise
iterations = 10


def D(x):
	# ---------
	# your code
	# ---------
	pass


def MatVec(x):
	# ---------
	# your code
	# ---------
	pass
	
def RMatVec(x):
	# ---------
	# your code
	# ---------
	pass


def shrink(a, k):
	# ---------
	# your code
	# ---------
	pass


def Optimization(b):	
	b = b.flatten()
	
	x = np.zeros((pix))
	z = np.zeros((pix*2))
	u = z
	
	for i in range(iterations):
		print(i)
		
		# x step:
		# ...
		
		# z step
		# ...
		
		# u step
		# ...
		
		
		print("Error: ", np.linalg.norm(img.flatten()-x))
	
	# Plot the results
	# ...
	
	return x.reshape(imgShape)
	
	

def EvaluateLinOp():
	# ---------
	# your code
	# ---------
	pass
	
	
def EvaluateShrink():
	# ---------
	# your code
	# ---------
	pass
	
	

# the main programm:
# ==================
	
# evaluation of tools
EvaluateLinOp()
EvaluateShrink()

# deblurring
plt.imsave("original.png", img)
blurred = convolve(img, C, 'same')
blurred += np.random.normal(0, sigma, blurred.shape)
plt.imsave("blurred.png", blurred)
x = Optimization(blurred)
plt.imsave("result.png", x)