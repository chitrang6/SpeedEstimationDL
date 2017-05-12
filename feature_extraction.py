# In this file, we are going to perform some of visulization techniques to find out 
# the difference between the two successive images. 

import cv2
from dataset_gen import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec




def crop_image(image_input):
	image_cropped = image_input[30:380, :]
	image = cv2.resize(image_cropped, (200, 66), interpolation = cv2.INTER_AREA)
	return image



def histogram_equ(image):
	img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img_output




def brightness_augmentation(image, factor):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	hsv_image[:,:,2] = hsv_image[:,:,2] * factor
	image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
	return image_rgb


def processing_image(image_path, ishistogramEquOn = True):
	image = cv2.imread(image_path)
	image = crop_image(image)
	if ishistogramEquOn:
		image  = histogram_equ(image)
	return image


def feature_diff(cur , next):
	diff = cv2.subtract(cur, next)
	return diff


def view_field():

	train , validation = dataset_gen()
	#print (train)
	# Here train is the panda frame and we will make new frame with the index
	train_visual = train.set_index("index")

	cur_image_path = train_visual.iloc[1][0]
	print train_visual.iloc[1]
	print (cur_image_path)
	speed = train_visual.iloc[1][2]
	print (speed)


#view_field()



# Let's first get the traiing and the validation data set and see what is saved in it


def visualize_data():

	train , validation = dataset_gen()

	#print (train)


	# Here train is the panda frame and we will make new frame with the index

	train_visual = train.set_index("index")




	for i in range(0, 20, 2):
		cur_image_path = train_visual.iloc[i][0]
		next_image_path = train_visual.iloc[i+1][0]

		fig = plt.figure()
		a=fig.add_subplot(1,3,1)


		cur_image = mpimg.imread(cur_image_path)
		cur_image  = histogram_equ(cur_image)
		cur_image = crop_image(cur_image)

		#cur_image = brightness_augmentation(cur_image , 0.5)
		cur_hsv = cv2.cvtColor(cur_image, cv2.COLOR_BGR2HSV)



		#cur_hsv = crop_image(cur_hsv)
		plt.imshow(cur_image)
		a.set_title('Current' + 'Shape:' + str(cur_image.shape))


		a=fig.add_subplot(1,3,2)

		next_image = mpimg.imread(next_image_path)
		next_image  = histogram_equ(next_image)
		#next_image = brightness_augmentation(next_image , 0.85)
		next_image = crop_image(next_image)
		next_hsv = cv2.cvtColor(next_image, cv2.COLOR_BGR2HSV)
		plt.imshow(next_image)
		a.set_title('Next' + 'Shape:' + str(next_image.shape))


		current = cv2.imread(cur_image_path)
		next_im = cv2.imread(next_image_path)

		current = crop_image(current)
		current= histogram_equ(current)
		next_im = crop_image(next_im)
		next_im= histogram_equ(next_im)

		
		diff = cv2.subtract(cur_hsv, next_hsv)
		rgb_diff = cv2.subtract(cur_image, next_image)

		diff1 = cv2.subtract(cur_hsv[:,:,0], next_hsv[:,:,0])
		diff2 = cv2.subtract(cur_hsv[:,:,1], next_hsv[:,:,1])
		diff3 = cv2.subtract(cur_hsv[:,:,2], next_hsv[:,:,2])

		a=fig.add_subplot(1,3,3)
		plt.imshow(rgb_diff)
		a.set_title('diff' + 'Shape:' + str(rgb_diff.shape))

		
		plt.show()


		num_row = 3
		num_col = 4
		gs = gridspec.GridSpec(num_row, num_col, top=1., bottom=0., right=1., left=0., hspace=0.,wspace=0.)
		ax = [plt.subplot(gs[i]) for i in range(num_row*num_col)]
		gs.update(hspace=0)
		
		ax[0].imshow(cur_hsv)
		ax[0].axis('off')

		ax[1].imshow(cur_hsv[:,:,0])
		ax[1].axis('off')

		ax[2].imshow(cur_hsv[:,:,1])
		ax[2].axis('off')

		ax[3].imshow(cur_hsv[:,:,2])
		ax[3].axis('off')


		ax[4].imshow(next_hsv)
		ax[4].axis('off')

		ax[5].imshow(next_hsv[:,:,0])
		ax[5].axis('off')

		ax[6].imshow(next_hsv[:,:,1])
		ax[6].axis('off')

		ax[7].imshow(next_hsv[:,:,2])
		ax[7].axis('off')


		ax[8].imshow(diff)
		ax[8].axis('off')
		ax[9].imshow(diff1)
		ax[9].axis('off')
		ax[10].imshow(diff2)
		ax[10].axis('off')
		ax[11].imshow(diff3)
		ax[11].axis('off')

		#hue, sat, val = cur_hsv[:,:,0], cur_hsv[:,:,1], cur_hsv[:,:,2]
		#plt.figure(figsize=(10,8))
		#plt.subplot(311)                             #plot in the first cell
		#plt.subplots_adjust(hspace=.5)
		#plt.title("Hue")
		#lt.hist(np.ndarray.flatten(hue), bins=180)
		#plt.subplot(312)                             #plot in the second cell
		#plt.title("Saturation")
		#plt.hist(np.ndarray.flatten(sat), bins=128)
		#plt.subplot(313)                             #plot in the third cell
		#plt.title("Luminosity Value")
		#plt.hist(np.ndarray.flatten(val), bins=128)
		#plt.show()

#visualize_data()

