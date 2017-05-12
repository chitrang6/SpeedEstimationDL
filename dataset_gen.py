import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt




def dataset_gen():
	current_path = os.getcwd()
	csv_file_path = current_path + "/speed_and_image.csv"

	df = pd.read_csv(csv_file_path)
	training_dataframe_path = current_path + "/train.csv"
	validation_dataframe_path = current_path + "/validation.csv"
	times = np.asarray(df['time'], dtype = np.float32)
	speeds = np.asarray(df['speed'], dtype=np.float32)

	#plt.plot(times, speeds, 'r-')
	##plt.title('For this video Speed of the Car v/s Time')
	#plt.show()


	#Now, the task is to create the Training and Validation data set.
	# Here to find the distinguable features we will take the current image and next image.
	# So, we will generate the data set according to it.

	trainng_data = pd.DataFrame()
	validation_data = pd.DataFrame()

	for i in xrange(len(df) -1):
		rowid1 = np.random.randint(len(df) - 1)
		rowid2 = rowid1 + 1

		row1 = df.iloc[[rowid1]].reset_index()
		row2 = df.iloc[[rowid2]].reset_index()

		# Here now we are going to spilt the validation and the training set. 
		rand_num = np.random.randint(9)
		if 0 <= rand_num <= 1:
			validation_frames = [validation_data, row1, row2]
			validation_data = pd.concat(validation_frames, axis = 0, join = 'outer', ignore_index=False)
		if rand_num >= 2:
			training_frames = [trainng_data, row1, row2]
			trainng_data = pd.concat(training_frames, axis = 0, join = 'outer', ignore_index=False)

	trainng_data.to_csv(training_dataframe_path)
	validation_data.to_csv(validation_dataframe_path)
	return trainng_data, validation_data

