import cv2
from dataset_gen import *
from feature_extraction import *


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def opticalFlowDense(image_current, image_next):
	gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
	gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
	hsv = np.zeros((66, 200	, 3))
	# set saturation
	#hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
	hsv[:, : ,1] = 255
	flow_mat = 0.5
	#image_scale = 1
	#nb_images = 1
	#win_size = 15
	#nb_iterations = 2
	#deg_expansion = 5
	#STD = 1
	#extra = 0
	#flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,flow_mat, image_scale, nb_images, win_size, nb_iterations, deg_expansion, STD, 0)
	flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next, flow_mat, 1, 3, 15, 3, 5, 1, 0)
	# convert from cartesian to polar
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
	# hue corresponds to direction
	hsv[:,:,0] = ang * (180/ np.pi / 2)
	# value corresponds to magnitude
	hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# convert HSV to int32's
	hsv = np.asarray(hsv, dtype= np.float32)
	rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	return rgb_flow
# Let's first get the traiing and the validation data set and see what is saved in it


train , validation = dataset_gen()


#print (train)


# Here train is the panda frame and we will make new frame with the index

train_visual = train.set_index("index")


for i in range(0, 50, 2):
	cur_image_path = train_visual.iloc[i][0]
	next_image_path = train_visual.iloc[i+1][0]
	current = cv2.imread(cur_image_path)
	next_im = cv2.imread(next_image_path)

	current = processing_image(cur_image_path , ishistogramEquOn = True)
	next_im = processing_image(next_image_path, ishistogramEquOn = True)

	
	optical_flow = opticalFlowDense(current, next_im)

	print (optical_flow.shape)
	plt.imshow(optical_flow)
	plt.show()