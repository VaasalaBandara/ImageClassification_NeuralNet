#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#loading the inbuilt fashion MNIST dataset
fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()


#exploring the training data
train_images.shape


#number of train labels
len(train_labels)

#the train labels
print(train_labels)


# # exploring testing data


test_images.shape

len(test_labels)


# # preprocessing data


#visualizing image using matplotlib
print("testing training image")
fig=plt.figure()
plt.imshow(train_images[0])
plt.colorbar()


# the values of color range from 0 to 255. they have to be adjusted to the range 0 to 1

train_images=train_images/255
test_images=test_images/255


# # building the neural network

# setting up layers



model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #shape of layer is 28 by 28
    tf.keras.layers.Dense(128, activation="relu"),#layer has 128 neurons densly connected
    tf.keras.layers.Dense(10)#layer has 10 neurons densly connected
])


# # compiling the model


model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# # training the model


model.fit(train_images,train_labels,epochs=10)


# # making predictions

# converting linear outputs of model to probabilities


probability_model=tf.keras.Sequential([model,
                                     tf.keras.layers.Softmax()])



predictions=probability_model.predict(test_images)


#testing an image
a=input("enter the image to be tested\n")
prediction=predictions[int(a)] #an array of probabilites
max_value=np.argmax(prediction) #maximum values of array of probabilities
print(max_value)

#the array of class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#making a plot to visualize prediciton
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots()
ax.imshow(test_images[int(a)])

#setting labels
label=class_names[max_value]
label2=class_names[test_labels[int(a)]]
xlabel=f"The predicted object is {label}"
title=f"the actual object is{label2}"
ax.set_xlabel(xlabel)
ax.set_title(title)
plt.show()

