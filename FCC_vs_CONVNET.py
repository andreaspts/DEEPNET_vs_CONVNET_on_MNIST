'''
Created on 12.03.2019

@author: Andreas

In this exercise we use Keras to analyze the MNIST dataset. 
To this aim, we use a simple fully connected neural network and compare it to a CONVNET approach.
For each unit (Preparation of the data, Defining and training the model, Evaluating the model and Presentation of the results)
we may repeat importing of packages for overview purposes.

Parts of the code are inspired by the book: Deep learning with python by F. Chollet, (Manning, 2018);

'''

# -------------- Preparation of the data for the FCC --------------

# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically

# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesFCC = train_images.reshape((60000, 28 * 28)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesFCC = train_imagesFCC.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]

test_imagesFCC = test_images.reshape((10000, 28 * 28)) # see above
test_imagesFCC = test_imagesFCC.astype('float32') / 255 # see above

train_labelsFCC = to_categorical(train_labels)
test_labelsFCC = to_categorical(test_labels)

#-------------- Plot a picture for illustration purposes from the training set ---------
# import matplotlib.pyplot as plt
# 
# image_index = 11111 # Select anything up to 60000
# print(train_labels[image_index])
# plt.imshow(train_images[image_index], cmap='Greys')
# plt.show()

# -------------- Preparation of the data for the CONVNET --------------
 
# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically
 
# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesCONVNET = train_images.reshape((60000, 28, 28, 1)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesCONVNET = train_imagesCONVNET.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]
 
test_imagesCONVNET = test_images.reshape((10000, 28, 28, 1)) # see above
test_images = test_imagesCONVNET.astype('float32') / 255 # see above
 
train_labelsCONVNET = to_categorical(train_labels)
test_labelsCONVNET = to_categorical(test_labels)
 
 
# # -------------- Defining and training the FCC model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from time import time
  
# Setting up the model architecture: A network with 2 fully connected layers with softmax output
modelFCC = models.Sequential()
modelFCC.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
modelFCC.add(layers.Dense(128, activation = 'relu'))
modelFCC.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelFCC.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelFCC.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
# # Tensorboard callback, here optional
# #callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
# #          write_graph=True, write_images=True,)]
  
# training the network via the fit method
historyFCC = modelFCC.fit(train_imagesFCC, train_labelsFCC, epochs = 10, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)
 
 
# -------------- Defining and training the CONVNET model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from time import time
 
# Setting up the model architecture: A CONVNET with maxpooling layers and 2 fully connected layers with softmax output
modelCONVNET = models.Sequential()
modelCONVNET.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)))
modelCONVNET.add(layers.MaxPooling2D((2,2)))
modelCONVNET.add(layers.Conv2D(64, (3,3),activation ='relu'))
modelCONVNET.add(layers.MaxPooling2D((2,2)))
modelCONVNET.add(layers.Conv2D(64, (3,3),activation = 'relu'))
modelCONVNET.add(layers.Flatten())
modelCONVNET.add(layers.Dense(64, activation = 'relu'))
modelCONVNET.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelCONVNET.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelCONVNET.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
# Tensorboard callback, here optional
#callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
#          write_graph=True, write_images=True,)]
 
# training the network via the fit method
historyCONVNET = modelCONVNET.fit(train_imagesCONVNET, train_labelsCONVNET, epochs = 10, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)
 
# -------------- Evaluating the FCC model --------------
# We are of course evaluating the model on the test set.
test_lossFCC, test_accFCC = modelFCC.evaluate(test_imagesFCC, test_labelsFCC)
print("The test accuracy of the FCC model amounts to: ",test_accFCC*100 ,"%")
 
# -------------- Evaluating the CONVNET model --------------
# We are of course evaluating the model on the test set.
test_lossCONVNET, test_accCONVNET = modelCONVNET.evaluate(test_imagesCONVNET, test_labelsCONVNET)
print("The test accuracy of the CONVNET model amounts to: ",test_accCONVNET*100 ,"%")
 
 
#------------------PRESENTATION OF RESULTS (PLOT OF ACCs)---------------
  
import matplotlib.pyplot as plt
  
accFCC = historyFCC.history['acc']
val_accFCC = historyFCC.history['val_acc']
lossFCC = historyFCC.history['loss']
val_lossFCC = historyFCC.history['val_loss']
  
accCONVNET = historyCONVNET.history['acc']
val_accCONVNET = historyCONVNET.history['val_acc']
lossCONVNET = historyCONVNET.history['loss']
val_lossCONVNET = historyCONVNET.history['val_loss']
  
# we run over the same number of epochs for both model
epochs = range(1, len(accCONVNET) + 1)
  
plt.plot(epochs, accFCC, '--ro', label='FCC Training acc')
plt.plot(epochs, val_accFCC,'--rx', label='FCC Validation acc')
plt.plot(epochs, accCONVNET, '--bo', label='CONVNET Training acc')
plt.plot(epochs, val_accCONVNET, '--bx', label='CONVNET Validation acc')
  
  
# plotting of training and validation accuracies
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.figure()
  
  
plt.plot(epochs, lossFCC, '--ro', label='FCC Training loss')
plt.plot(epochs, val_lossFCC, '--rx', label='FCC Validation loss')
plt.plot(epochs, lossCONVNET, '--bo', label='CONVNET Training loss')
plt.plot(epochs, val_lossCONVNET, '--bx', label='CONVNET Validation loss')
  
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
# pred = modelCONVNET.predict(test_images[image_index].reshape(1, 28, 28, 1))
# print("The convnet predicts the number to be: ",pred.argmax(),".")

# import matplotlib.pyplot as plt
# image_index = 9999
# plt.imshow(test_images[image_index].reshape(28, 28),cmap='Greys')
# plt.show()
# pred = modelFCC.predict(test_images[image_index].reshape(1, 28, 28, 1))
# print("The FCC predicts the number to be: ",pred.argmax(),".")

#--------------------------------------------------------
