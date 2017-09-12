import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve, correlate
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsmr

plt.close('all')


# load image, crop it and take the red channel:
img = plt.imread("Lenna.png")[200:300,200:320,0]
# img = plt.imread("Lenna.png")[250:254,250:254,0]
imgShape = img.shape
pix = imgShape[0]*imgShape[1] # number of pixels

# kernels
# ---------
# your code
# C = ...
# d_x = ...
# d_y = ...
# ---------

def gkern(size=3, sigma=1.):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

l = 0.001
rho = 0.1 # strength of regularization
sigma = 0.01 # the amount of noise
iterations = 10

kernel = gkern(size=3, sigma=1)
# kernel = np.ones((3,3))
# print(C)
d_x = np.array([[0,0,0],[0,-1,1],[0,0,0]])
d_y = np.array([[0,0,0],[0,-1,0],[0,1,0]])


def C(x):
	# assume x given in imgShape
	# kernel = gkern(size=9, sigma=1)
	blur = convolve(x, kernel, 'same')
	return blur.flatten()

def D(x):
	# ---------
	# your code
	# ---------
	# assume x given in imgShape
	delta_x = convolve(x, d_x, 'same')
	delta_y = convolve(x, d_y, 'same')
	return np.hstack([delta_x.flatten(), delta_y.flatten()])

def MatVec(x):
	# ---------
	# your code
	# ---------
	y = x.reshape(imgShape)
	return np.hstack([C(y), rho*D(y)])
	
def RMatVec(x):
	# ---------
	# your code
	# ---------
	y1 = x[:pix].reshape(imgShape)
	y2 = x[pix:2*pix].reshape(imgShape)
	y3 = x[2*pix:3*pix].reshape(imgShape)

	C_t = correlate(y1, kernel, 'same').flatten()
	delta_x_t = correlate(y2, rho*d_x, 'same').flatten()
	delta_y_t = correlate(y3, rho*d_y, 'same').flatten()

	return C_t + delta_x_t + delta_y_t


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

	f = np.ones(pix)
	h = np.ones(3*pix)
	rhs1 = MatVec(f)
	rhs2 = RMatVec(h)
	
	val1 = rhs1.dot(h)
	val2 = rhs2.dot(f)
	print("val1 = {}\nval2 = {}".format(val1, val2))
	assert (val1-val2 <= 1.e-5)
	
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

# # deblurring
plt.imsave("original.png", img)
blurred = convolve(img, C, 'same')
blurred += np.random.normal(0, sigma, blurred.shape)
plt.imsave("blurred.png", blurred)
x = Optimization(blurred)
plt.imsave("result.png", x)