'''
Created on 12.03.2019

@author: Andreas

In this exercise we use Keras to analyze the MNIST dataset. 
To this aim, we use two simple fully connected neural networks and compare their performance.
For each unit (Preparation of the data, Defining and training the model, Evaluating the model and Presentation of the results)
we may repeat importing of packages for overview purposes.

Parts of the code are inspired by the book: Deep learning with python by F. Chollet, (Manning, 2018);

'''

# -------------- Preparation of the data for the FCC1 --------------

# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically

# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesFCC1 = train_images.reshape((60000, 28 * 28)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesFCC1 = train_imagesFCC1.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]

test_imagesFCC1 = test_images.reshape((10000, 28 * 28)) # see above
test_imagesFCC1 = test_imagesFCC1.astype('float32') / 255 # see above

train_labelsFCC1 = to_categorical(train_labels)
test_labelsFCC1 = to_categorical(test_labels)

#-------------- Plot a picture for illustration purposes from the training set ---------
# import matplotlib.pyplot as plt
# 
# image_index = 11111 # Select anything up to 60000
# print(train_labels[image_index])
# plt.imshow(train_images[image_index], cmap='Greys')
# plt.show()

# -------------- Preparation of the data for the FCC2 --------------

# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically

# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesFCC2 = train_images.reshape((60000, 28 * 28)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesFCC2 = train_imagesFCC2.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]

test_imagesFCC2 = test_images.reshape((10000, 28 * 28)) # see above
test_imagesFCC2 = test_imagesFCC2.astype('float32') / 255 # see above

train_labelsFCC2 = to_categorical(train_labels)
test_labelsFCC2 = to_categorical(test_labels)
 
 
# # -------------- Defining and training the FCC1 model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from time import time
  
# Setting up the model architecture: A network with 2 fully connected layers with softmax output
modelFCC1 = models.Sequential()
modelFCC1.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
modelFCC1.add(layers.Dense(128, activation = 'relu'))
modelFCC1.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelFCC1.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelFCC1.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
# # Tensorboard callback, here optional
# #callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
# #          write_graph=True, write_images=True,)]
  
# training the network via the fit method
historyFCC1 = modelFCC1.fit(train_imagesFCC1, train_labelsFCC1, epochs = 30, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)
 
 
# # -------------- Defining and training the FCC2 model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from time import time
from keras import regularizers
  
# Setting up the model architecture: A network with 2 fully connected layers with softmax output
modelFCC2 = models.Sequential()
modelFCC2.add(layers.Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001), input_shape = (28 * 28,)))
modelFCC2.add(layers.Dropout(0.1))
modelFCC2.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
#modelFCC2.add(layers.Dropout(0.2))
modelFCC2.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelFCC2.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelFCC2.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
# # Tensorboard callback, here optional
# #callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
# #          write_graph=True, write_images=True,)]
  
# training the network via the fit method
historyFCC2 = modelFCC2.fit(train_imagesFCC2, train_labelsFCC2, epochs = 30, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)
 
 
# -------------- Evaluating the FCC1 model --------------
# We are of course evaluating the model on the test set.
test_lossFCC1, test_accFCC1 = modelFCC1.evaluate(test_imagesFCC1, test_labelsFCC1)
print("The test accuracy of the FCC1 model amounts to: ",test_accFCC1*100 ,"%")
 
# -------------- Evaluating the FCC2 model --------------
# We are of course evaluating the model on the test set.
test_lossFCC2, test_accFCC2 = modelFCC2.evaluate(test_imagesFCC2, test_labelsFCC2)
print("The test accuracy of the FCC2 model amounts to: ",test_accFCC2*100 ,"%")
 
 
#------------------PRESENTATION OF RESULTS (PLOT OF ACCs)---------------
  
import matplotlib.pyplot as plt
  
accFCC1 = historyFCC1.history['acc']
val_accFCC1 = historyFCC1.history['val_acc']
lossFCC1 = historyFCC1.history['loss']
val_lossFCC1 = historyFCC1.history['val_loss']
  
accFCC2 = historyFCC2.history['acc']
val_accFCC2 = historyFCC2.history['val_acc']
lossFCC2 = historyFCC2.history['loss']
val_lossFCC2 = historyFCC2.history['val_loss']
  
# we run over the same number of epochs for both model
epochs = range(1, len(accFCC1) + 1)
  
plt.plot(epochs, accFCC1, '--ro', label='FCC1 Training acc')
plt.plot(epochs, val_accFCC1,'--rx', label='FCC1 Validation acc')
plt.plot(epochs, accFCC2, '--bo', label='FCC2 (regularized) Training acc')
plt.plot(epochs, val_accFCC2, '--bx', label='FCC2 (regularized) Validation acc')
  
  
# plotting of training and validation accuracies
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.figure()
  
  
plt.plot(epochs, lossFCC1, '--ro', label='FCC1 Training loss')
plt.plot(epochs, val_lossFCC1, '--rx', label='FCC1 Validation loss')
plt.plot(epochs, lossFCC2, '--bo', label='FCC2 (regularized) Training loss')
plt.plot(epochs, val_lossFCC2, '--bx', label='FCC2 (regularized) Validation loss')
  
# plotting of training and validation losses
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
  
#------------------------- Check the prediction of the networks -------------------------------

# import matplotlib.pyplot as plt
# image_index = 9999
# plt.imshow(test_images[image_index].reshape(28, 28),cmap='Greys')
# plt.show()
# pred = modelFCC2.predict(test_images[image_index].reshape(1, 28, 28, 1))
# print("The convnet predicts the number to be: ",pred.argmax(),".")

# import matplotlib.pyplot as plt
# image_index = 9999
# plt.imshow(test_images[image_index].reshape(28, 28),cmap='Greys')
# plt.show()
# pred = modelFCC.predict(test_images[image_index].reshape(1, 28, 28, 1))
# print("The FCC predicts the number to be: ",pred.argmax(),".")

#--------------------------------------------------------
