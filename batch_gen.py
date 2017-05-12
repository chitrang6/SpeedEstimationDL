from feature_extraction import *
import numpy as np
from sklearn.utils import shuffle


def generate_training_data(data, batch_size = 32):
    """This function will be useful to generate the batch of the input data"""



    image = np.zeros((batch_size, 66, 200, 3))
    label = np.zeros((batch_size))
    

    new_data = data.set_index("index")
    while True:
        for i in range(batch_size):
            num_gen = np.random.randint(1, len(new_data) - 1)

            row_cur = new_data.iloc[num_gen]
            row_next = new_data.iloc[num_gen+1]   
            cur_image = processing_image(row_cur[0], ishistogramEquOn = True)
            next_image = processing_image(row_next[0], ishistogramEquOn = True)


            rgb_diff = feature_diff(cur_image, next_image)
            cur_speed =  row_cur[2]   
            next_speed =  row_next[2]    
            avg_speed = np.mean([cur_speed, next_speed])
            
            image[i] = rgb_diff
            label[i] = avg_speed
        yield shuffle(image, label)








def generate_validation_data(data):
    """ THis function will generate the validation data.  """
    new_data = data.set_index("index")
    while True:
        for i in range(1, len(new_data) - 1):
            row_cur = new_data.iloc[i]
            row_next = new_data.iloc[i+1]
            cur_image = processing_image(row_cur[0], ishistogramEquOn = True)
            next_image = processing_image(row_next[0], ishistogramEquOn = True)


            rgb_diff = feature_diff(cur_image, next_image)
            rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0] , rgb_diff.shape[1] , rgb_diff.shape[2])
            cur_speed =  row_cur[2]      
            next_speed =  row_next[2]      
            avg_speed = np.mean([cur_speed, next_speed])
            speed = np.array([[avg_speed]])

            yield rgb_diff, speed


