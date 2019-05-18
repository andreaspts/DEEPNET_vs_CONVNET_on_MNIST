'''
Created on 12.03.2019

@author: Andreas

In this exercise we use Keras to analyze the MNIST dataset. 
To this aim, we use two CONVNETs and compare the results.
For each unit (Preparation of the data, Defining and training the model, Evaluating the model and Presentation of the results)
we may repeat importing of packages for overview purposes.

Parts of the code are inspired by the book: Deep learning with python by F. Chollet, (Manning, 2018);

'''

# -------------- Preparation of the data for the CONVNET1 --------------

# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically

# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesCONVNET1 = train_images.reshape((60000, 28, 28, 1)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesCONVNET1 = train_imagesCONVNET1.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]

test_imagesCONVNET1 = test_images.reshape((10000, 28, 28, 1)) # see above
test_images = test_imagesCONVNET1.astype('float32') / 255 # see above

train_labelsCONVNET1 = to_categorical(train_labels)
test_labelsCONVNET1 = to_categorical(test_labels)


# -------------- Preparation of the data for the CONVNET2 --------------

# We initialize keras and mount the required packages, e.g. the data set itself
import keras
from keras.datasets import mnist
from keras.utils import to_categorical # needed to encode the labels categorically

# We get the training images and labels as well as the test images and labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_imagesCONVNET2 = train_images.reshape((60000, 28, 28, 1)) # images are stored in an array of shape (60000, 28, 28) of type unit8 with values in the interval [0,255] 
train_imagesCONVNET2 = train_imagesCONVNET2.astype('float32') / 255 # transforming the array to type float32 with values in the interval to [0,1]

test_imagesCONVNET2 = test_images.reshape((10000, 28, 28, 1)) # see above
test_images = test_imagesCONVNET2.astype('float32') / 255 # see above

train_labelsCONVNET2 = to_categorical(train_labels)
test_labelsCONVNET2 = to_categorical(test_labels)


# -------------- Defining and training the CONVNET1 model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from time import time
 
# Setting up the model architecture: A CONVNET with maxpooling layers and 2 fully connected layers with softmax output
modelCONVNET1 = models.Sequential()
modelCONVNET1.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)))
modelCONVNET1.add(layers.MaxPooling2D((2,2)))
modelCONVNET1.add(layers.Conv2D(64, (3,3),activation ='relu'))
modelCONVNET1.add(layers.MaxPooling2D((2,2)))
modelCONVNET1.add(layers.Conv2D(64, (3,3),activation = 'relu'))
modelCONVNET1.add(layers.Flatten())
modelCONVNET1.add(layers.Dense(64, activation = 'relu'))
modelCONVNET1.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelCONVNET1.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelCONVNET1.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
# Tensorboard callback, here optional
#callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
#          write_graph=True, write_images=True,)]
 
# training the network via the fit method
historyCONVNET1 = modelCONVNET1.fit(train_imagesCONVNET1, train_labelsCONVNET1, epochs = 30, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)


# -------------- Defining and training the CONVNET2 model --------------
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras import regularizers
from keras.callbacks import TensorBoard
from time import time

# Setting up the model architecture: A CONVNET with maxpooling layers and 2 fully connected layers with softmax output
modelCONVNET2 = models.Sequential()
modelCONVNET2.add(layers.Conv2D(32,(3,3), input_shape = (28,28,1)))

modelCONVNET2.add(layers.BatchNormalization())
modelCONVNET2.add(layers.Activation("relu"))

modelCONVNET2.add(layers.MaxPooling2D((2,2)))
modelCONVNET2.add(layers.Conv2D(64, (3,3)))#,activation ='relu'))

modelCONVNET2.add(layers.BatchNormalization())
modelCONVNET2.add(layers.Activation("relu"))


modelCONVNET2.add(layers.MaxPooling2D((2,2)))
modelCONVNET2.add(layers.Conv2D(64, (3,3)))#,activation ='relu'))

modelCONVNET2.add(layers.BatchNormalization())
modelCONVNET2.add(layers.Activation("relu"))
modelCONVNET2.add(layers.Flatten())
modelCONVNET2.add(layers.Dense(64, activation = 'relu')) #kernel_regularizer=regularizers.l2(0.001),
#modelCONVNET2.add(layers.Dropout(0.2))
modelCONVNET2.add(layers.Dense(10,activation='softmax'))
# Print the model architecture
modelCONVNET2.summary()
# Compilation of the mode: adding optimizer and loss for back propagation
# loss function: specifies how the network measures its performance on the training set
# optimizer: providing a mechanism by which the network will update itself based on the data it is fed with and the loss function
# another optimizer could be the rmsprop
# accuracy metric: to monitor the fraction of images which were correctly classified
modelCONVNET2.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Tensorboard callback, here optional
#callbacks = [TensorBoard(log_dir = 'logs/{}'.format(time()),histogram_freq=1,  
#          write_graph=True, write_images=True,)]

# training the network via the fit method
historyCONVNET2 = modelCONVNET2.fit(train_imagesCONVNET2, train_labelsCONVNET2, epochs = 30, batch_size = 64, validation_split=0.1)#, callbacks=callbacks)

# -------------- Evaluating the CONVNET1 model --------------
# We are of course evaluating the model on the test set.
test_lossCONVNET1, test_accCONVNET1 = modelCONVNET1.evaluate(test_imagesCONVNET1, test_labelsCONVNET1)
print("The test accuracy of the CONVNET1 model amounts to: ",test_accCONVNET1*100 ,"%")

# -------------- Evaluating the CONVNET2 model --------------
# We are of course evaluating the model on the test set.
test_lossCONVNET2, test_accCONVNET2 = modelCONVNET2.evaluate(test_imagesCONVNET2, test_labelsCONVNET2)
print("The test accuracy of the CONVNET2 model amounts to: ",test_accCONVNET2*100 ,"%")


#------------------PRESENTATION OF RESULTS (PLOT OF ACCs)---------------

import matplotlib.pyplot as plt

accCONVNET1 = historyCONVNET1.history['acc']
val_accCONVNET1 = historyCONVNET1.history['val_acc']
lossCONVNET1 = historyCONVNET1.history['loss']
val_lossCONVNET1 = historyCONVNET1.history['val_loss']

accCONVNET2 = historyCONVNET2.history['acc']
val_accCONVNET2 = historyCONVNET2.history['val_acc']
lossCONVNET2 = historyCONVNET2.history['loss']
val_lossCONVNET2 = historyCONVNET2.history['val_loss']

# we run over the same number of epochs for both model
epochs = range(1, len(accCONVNET1) + 1)

plt.plot(epochs, accCONVNET1, '--ro', label='CONVNET1 Training acc')
plt.plot(epochs, val_accCONVNET1,'--rx', label='CONVNET1 Validation acc')
plt.plot(epochs, accCONVNET2, '--bo', label='CONVNET2 (Batch Normalization) Training acc')
plt.plot(epochs, val_accCONVNET2, '--bx', label='CONVNET2 (Batch Normalization) Validation acc')


# plotting of training and validation accuracies
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.figure()


plt.plot(epochs, lossCONVNET1, '--ro', label='CONVNET1 Training loss')
plt.plot(epochs, val_lossCONVNET1, '--rx', label='CONVNET1 Validation loss')
plt.plot(epochs, lossCONVNET2, '--bo', label='CONVNET2 (Batch Normalization) Training loss')
plt.plot(epochs, val_lossCONVNET2, '--bx', label='CONVNET2 (Batch Normalization) Validation loss')

# plotting of training and validation losses
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#--------------------------------------------------------

