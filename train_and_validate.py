from batch_gen import *

from nvidia_speed_model import *
from dataset_gen import *

train , validation = dataset_gen()

BATCH_SIZE = 16

# First We will generate the traiing and validation data generator 

training_data_gen = generate_training_data(train , BATCH_SIZE)

validation_data_gen = generate_validation_data(validation)




from keras.callbacks import EarlyStopping, ModelCheckpoint


current_path = os.getcwd()
filepath = current_path + 'nvidia_speed_model.h5'

earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=1, 
                              verbose=1, 
                              min_delta = 0.23,
                              mode='min',)
modelCheckpoint = ModelCheckpoint(filepath, 
                                  monitor = 'val_loss', 
                                  save_best_only = True, 
                                  mode = 'min', 
                                  verbose = 1,
                                 save_weights_only = True)
callbacks_list = [modelCheckpoint, earlyStopping]

train_size = train.shape[0]

validation_size = validation.shape[0]

print("Training Data size: " + str(train_size) + " Validation size: " + str(validation_size))


model = nvidia_speed_model()

history = model.fit_generator(
        training_data_gen, 
        steps_per_epoch = 200, 
        epochs = 20,
    	callbacks = callbacks_list,
        verbose = 1,
        validation_data = validation_data_gen,
        validation_steps = validation_size)

print(history)